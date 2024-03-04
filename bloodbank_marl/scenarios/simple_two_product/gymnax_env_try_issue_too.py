import gymnax
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Union, Optional, List, Dict
from functools import partial
from flax import struct
from jax import lax
import distrax
import matplotlib.pyplot as plt
import numpy as np

# TODO: KPIs talk about blood groups, but this is more generic. Doesn't matter for now but could update.

# TODO: Fill out missing methods (e.g. for action/obs/state spaces etc) if we end up using this for final results

# TODO: Decide what to do about EnvObs and policies - can write specific policies for the single
# agent env, do what we currently do here, or rethink the use of them if it is what is slowing down
# the multiagent env.

# TODO THink about best way to have default issuing policy, and use config to specify another
# exact_match = FixedPolicy(exact_match_policy, None, {})

# TODO: Decide where to specify n_products and how to get it into default params

n_products = 2
C = 1e10  # invalid substitution cost
max_useful_life = 2
substitution_cost_ratios = [
    [
        0,
        C,
    ],
    [
        1,
        0,
    ],
]

product_probabilities = [0.50, 0.50]

action_mask_per_request_type = np.array([[1, 0], [1, 1]])


@struct.dataclass
class EnvState:
    stock: chex.Array
    in_transit: chex.Array
    step: int  # For the purposes of this env, a step is a day
    issue_policy_params: Optional[Dict] = None


@struct.dataclass
class EnvParams:
    poisson_demand_mean: chex.Array
    product_probabilities: chex.Array
    age_on_arrival_distribution_probs: chex.Array
    fixed_order_cost: float
    variable_order_costs: chex.Array
    shortage_costs: chex.Array
    wastage_costs: chex.Array
    holding_costs: chex.Array
    substitution_costs: chex.Array
    action_mask_per_request_type: chex.Array
    max_expiry_pc_target: float
    min_service_level_pc_target: float
    min_exact_match_pc_target: float
    target_kpi_breach_penalty: float
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        cls,
        poisson_demand_mean: int = 4,
        product_probabilities: List[float] = product_probabilities,
        age_on_arrival_distribution_probs: List[float] = [1]
        + [0] * (max_useful_life - 1),
        fixed_order_cost: float = 0,
        variable_order_costs: List[float] = [3.0, 3.0],
        shortage_costs: List[float] = [5.0, 5.0],
        wastage_costs: List[float] = [7.0, 7.0],
        holding_costs: List[float] = [1.0, 1.0],
        substitution_cost_ratios: List[List[float]] = substitution_cost_ratios,
        max_substitution_cost: float = 2.0,
        action_mask_per_request_type: List[int] = action_mask_per_request_type,
        max_expiry_pc_target: float = 100.0,  # Effectively no limit by default
        min_service_level_pc_target: float = 0.0,  # Effectively no limit by default
        min_exact_match_pc_target: float = 0.0,  # Effectively no limit by default
        target_kpi_breach_penalty: float = 0.0,  # Set penalty to 0 for now, was having issue with this
        max_steps_in_episode: int = 365,  # By default, we run for a year
        gamma: float = 1.0,
    ):
        return cls(
            jnp.array(poisson_demand_mean),
            jnp.array(product_probabilities),
            jnp.array(age_on_arrival_distribution_probs),
            fixed_order_cost,
            jnp.array(variable_order_costs),
            jnp.array(shortage_costs),
            jnp.array(wastage_costs),
            jnp.array(holding_costs),
            jnp.array(substitution_cost_ratios) * max_substitution_cost,
            jnp.array(action_mask_per_request_type),
            max_expiry_pc_target,
            min_service_level_pc_target,
            min_exact_match_pc_target,
            target_kpi_breach_penalty,
            max_steps_in_episode,
            gamma,
        )


@struct.dataclass
class EnvObs:
    stock: chex.Array
    in_transit: chex.Array
    action_mask: chex.Array

    @property
    def obs(self):
        batch_dims = self.in_transit.shape[:-2]
        return jnp.hstack(
            [
                self.in_transit.reshape(batch_dims + (-1,)),
                self.stock.reshape(batch_dims + (-1,)),
            ]
        )

    @property
    def obs_total_per_product(self):
        batch_dims = self.in_transit.shape[:-2]
        inv = self.stock.sum(axis=-1) + self.in_transit.sum(axis=-1)
        return jnp.hstack(
            [
                inv.reshape(batch_dims + (-1,)),
            ]
        )


@struct.dataclass
class DemandInfo:
    total_demand: int
    remaining_demand: int
    shortages: int
    allocations: int
    remaining_stock: chex.Array
    in_transit: chex.Array
    request_type_samples: chex.Array
    key: jax.random.PRNGKey
    issue_policy_params: Optional[Dict]
    action_mask_per_request_type: chex.Array


@struct.dataclass
class IssueObs:
    stock: chex.Array
    in_transit: chex.Array
    request_type: chex.Array
    action_mask: chex.Array

    @property
    def obs(self):
        batch_dims = self.in_transit.shape[:-2]
        return jnp.hstack(
            [
                self.request_type.reshape(batch_dims + (-1,)),
                self.in_transit.reshape(batch_dims + (-1,)),
                self.stock.reshape(batch_dims + (-1,)),
            ]
        )


class SimpleTwoProductPerishableIncIssueGymnax(environment.Environment):
    """Jittable abstract base class for all gymnax Environments."""

    def __init__(
        self,
        n_products: int = 2,
        max_useful_life: int = 2,
        lead_time: int = 1,
        max_order_quantity: int = 10,  # Applies to any individual product, rather than in total
        max_demand: int = 100,  # TODO: Check older work
    ):
        super().__init__()
        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantities = jnp.array([max_order_quantity] * self.n_products)
        self.max_demand = max_demand

    def set_issuing_fn(self, issuing_fn):
        self._issuing_fn = issuing_fn

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(
            key,
            state,
            action,
            params,
        )
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        # This is to handle keeping issue policy params over episode boundaries
        state = state.replace(issue_policy_params=state_st.issue_policy_params)
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Environment-specific step transition."""
        cumulative_gamma = self.cumulative_gamma(state, params)
        info = {}
        info["cumulative_gamma"] = cumulative_gamma
        stock, in_transit = state.stock, state.in_transit

        interval_key, type_key, arrival_key = jax.random.split(key, 3)
        # Add the new order to in_transit
        orders = jnp.clip(action, 0, self.max_order_quantities)
        order_placed = jax.lax.cond(jnp.sum(orders) > 0, lambda: 1, lambda: 0)
        info["orders"] = orders
        # TODO: Handle cases where lead time = 0
        in_transit = in_transit.at[0 : self.n_products, 0].set(orders)

        # If the lead time is 0, then receive order immediately; otherwise add to in transit
        stock, in_transit = jax.lax.cond(
            self.lead_time == 0,
            self._receive_order,
            lambda arrival_key, stock, in_transit, params: (stock, in_transit),
            arrival_key,
            stock,
            in_transit,
            params,
        )

        # Calculate fixed and variable order cost
        variable_order_cost = jnp.dot(-params.variable_order_costs, orders)
        fixed_order_cost = -params.fixed_order_cost * order_placed

        # Sample total demand for the day
        request_intervals = distrax.Gamma(
            concentration=1, rate=params.poisson_demand_mean
        ).sample(seed=interval_key, sample_shape=(self.max_demand,))
        cum_time = jnp.cumsum(request_intervals)
        remaining_demand = jnp.sum(jnp.where(cum_time < 1, 1, 0))

        # Sample the product types for demand
        request_types = distrax.Categorical(probs=params.product_probabilities).sample(
            seed=type_key, sample_shape=(self.max_demand,)
        )
        # Construct a demand info object
        allocations = jnp.zeros(
            (self.n_products, self.n_products, self.max_useful_life)
        )  # TODO: Check dimensions
        shortages = jnp.zeros((self.n_products,))
        demand_info = DemandInfo(
            remaining_demand,
            remaining_demand,
            shortages,
            allocations,
            stock,
            in_transit,
            request_types,
            arrival_key,
            state.issue_policy_params,
            params.action_mask_per_request_type,
        )

        # Fill demand as far as possible
        # Calculate shortage and substitution costs
        stock, shortages, allocations = self._fill_demand(demand_info)
        shortage_cost = jnp.dot(-params.shortage_costs, shortages)
        substitution_cost = (
            -params.substitution_costs * allocations.sum(axis=-1)
        ).sum()  # Sum allocations over ages because sub cost just based on types
        info["shortages"] = shortages
        info["allocations"] = allocations
        info["demand"] = shortages + allocations.sum(axis=(-2, -1))

        # Age stock and calculate expiries
        # Calculate wastage costs
        stock, expiries = self._age_stock(stock)
        holding = stock.sum(axis=-1)
        wastage_cost = jnp.dot(-params.wastage_costs, expiries)
        holding_cost = jnp.dot(-params.holding_costs, holding)
        info["holding"] = holding
        info["expiries"] = expiries

        # Calculate the reward
        reward = (
            variable_order_cost
            + fixed_order_cost
            + shortage_cost
            + substitution_cost
            + wastage_cost
            + holding_cost
        )

        info["variable_order_cost"] = variable_order_cost
        info["fixed_order_cost"] = fixed_order_cost
        info["shortage_cost"] = shortage_cost
        info["substitution_cost"] = substitution_cost
        info["wastage_cost"] = wastage_cost
        info["holding_cost"] = holding_cost

        # Receive the order placed L-1 periods ago (i.e. start of this step when L=1) if lead time is >= 1
        # As part of this, sample from distribution of remaining useful life on arrival
        stock, in_transit = jax.lax.cond(
            self.lead_time > 0,
            self._receive_order,
            lambda arrival_key, stock, in_transit, params: (stock, in_transit),
            arrival_key,
            stock,
            in_transit,
            params,
        )

        # Update the state
        state = EnvState(
            stock=stock,
            in_transit=in_transit,
            step=state.step + 1,
            issue_policy_params=state.issue_policy_params,
        )  # TODO Add in the other elements
        done = self.is_terminal(state, params)

        info["day_counter"] = 1  # Used when we are accumulating infos for KPIs

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        # When lead time is 0, still have one slot to put order into and then immediately remove from for simplicity
        # when using receive order functions
        state = EnvState(
            stock=jnp.zeros((self.n_products, self.max_useful_life)),
            in_transit=jnp.zeros((self.n_products, max(self.lead_time, 1))),
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""

        # Simple action masking, for now can always order each product

        # NOTE: For L = 0, observation made at the end of the day after ageing stock, and first element would always be zero
        # In older work we removed the first element, but we need to keep it in the multi-agent environment for issuing steps
        # So keep here too for consistency of states between the two versions (e.g. means we should be able to use same NN
        # replenishment policy on both)
        return EnvObs(
            stock=state.stock,
            # If lead time is 0 or 1, nothing to include in obs
            in_transit=state.in_transit,
            action_mask=jnp.ones((self.n_products,)),
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""
        done_steps = state.step >= params.max_steps_in_episode
        return done_steps

    def cumulative_gamma(self, state: EnvState, params: EnvParams) -> float:
        """Return cumulative discount factor"""
        return params.gamma**state.step

    @property
    def name(self) -> str:
        """Environment name."""
        return "SimpleTwoProductPerishableIncIssueGymnax"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.n_products

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        raise NotImplementedError

    def _fill_demand(
        self, initial_demand_info: DemandInfo
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Fill demand as far as possible, returning the remaining demand, shortages and allocations"""
        demand_info = jax.lax.while_loop(
            self._remaining_demand, self._fill_one_request, initial_demand_info
        )
        return (
            demand_info.remaining_stock,
            demand_info.shortages,
            demand_info.allocations,
        )

    def _remaining_demand(self, demand_info: DemandInfo):
        return demand_info.remaining_demand > 0

    def _fill_one_request(self, demand_info: DemandInfo) -> DemandInfo:
        idx = demand_info.total_demand - demand_info.remaining_demand
        key, issue_key = jax.random.split(demand_info.key)
        remaining_demand = demand_info.remaining_demand - 1
        requested_product_idx = demand_info.request_type_samples[idx]
        # Identify the product type to be issued
        issuing_obs = IssueObs(
            stock=demand_info.remaining_stock,
            in_transit=demand_info.in_transit,
            request_type=requested_product_idx,
            action_mask=self._get_issuing_mask(demand_info, requested_product_idx),
        )
        issue_action = self._issuing_fn(
            demand_info.issue_policy_params, issuing_obs, issue_key
        )
        issued_product_idx = jnp.argmax(issue_action)
        shortage = jax.lax.select(
            jax.lax.bitwise_or(
                jnp.sum(issue_action) == 0,
                demand_info.remaining_stock[
                    issued_product_idx, 0 : self.max_useful_life
                ].sum()
                == 0,
            ),
            1,
            0,
        )
        stock_after_issue = jax.lax.cond(
            shortage < 1,
            self._issue_one_unit,
            lambda stock, issued_product_idx: stock,
            demand_info.remaining_stock,
            issued_product_idx,
        )
        issued = demand_info.remaining_stock - stock_after_issue
        allocations = demand_info.allocations.at[requested_product_idx, :, :].add(
            issued
        )
        shortages = demand_info.shortages.at[requested_product_idx].add(shortage)
        return DemandInfo(
            demand_info.total_demand,
            remaining_demand,
            shortages,
            allocations,
            stock_after_issue,
            demand_info.in_transit,
            demand_info.request_type_samples,
            key,
            demand_info.issue_policy_params,
            demand_info.action_mask_per_request_type,
        )

    def _get_issuing_mask(
        self, demand_info: DemandInfo, requested_product_idx: int
    ) -> chex.Array:
        return (
            jnp.where(demand_info.remaining_stock.sum(axis=-1) > 0, 1, 0)
            * demand_info.action_mask_per_request_type[requested_product_idx]
        )

    def _issue_one_unit(self, stock: chex.Array, product_idx: int) -> chex.Array:
        return stock.at[product_idx].set(self._issue_fifo(stock[product_idx]))

    def _issue_fifo(self, stock: chex.Array) -> chex.Array:
        """Issue stock using FIFO policy"""
        age_idx = (self.max_useful_life - 1) - (stock[::-1] > 0).argmax()
        return jnp.clip(stock.at[age_idx].add(-1), a_min=0)

    def _age_stock(self, stock: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Age stock by one period and calculate expiries"""
        expiries = stock[: self.n_products, self.max_useful_life - 1]
        stock = jnp.roll(stock, axis=1, shift=1)
        stock = stock.at[: self.n_products, 0].set(0)
        return stock, expiries

    def _receive_order(
        self,
        key: chex.PRNGKey,
        stock: chex.Array,
        in_transit: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, chex.Array]:
        """Receive an order that was placed L-1 periods ago"""
        stock_received = self._sample_ages_on_arrival(
            key,
            params.age_on_arrival_distribution_probs,
            in_transit[: self.n_products, -1],
        )
        stock = stock + stock_received
        in_transit = jnp.roll(in_transit, axis=1, shift=1)
        in_transit = in_transit.at[: self.n_products, 0].set(0)
        return stock, in_transit

    def _sample_ages_on_arrival(
        self,
        key: chex.PRNGKey,
        age_on_arrival_distribution_probs: chex.Array,
        order_received: chex.Array,
    ) -> chex.Array:
        # The nice thing about this is that if we have different probs for each
        # product, we can feed them in an an array (n_prods x max_useful_life) and
        # this will still work
        d = distrax.Multinomial(
            total_count=order_received, probs=age_on_arrival_distribution_probs
        )
        return d.sample(seed=key)

    @property
    def empty_infos(self) -> Dict[str, Union[chex.Array, float, int]]:
        return {
            "allocations": jnp.zeros(
                (self.n_products, self.n_products, self.max_useful_life)
            ),
            "cumulative_gamma": 1.0,
            "day_counter": 0,
            # "reward": 0.0,
            "demand": jnp.zeros((self.n_products,)),
            "expiries": jnp.zeros((self.n_products,)),
            "holding": jnp.zeros((self.n_products,)),
            "orders": jnp.zeros((self.n_products,)),
            "shortages": jnp.zeros((self.n_products,)),
            "variable_order_cost": 0.0,
            "fixed_order_cost": 0.0,
            "shortage_cost": 0.0,
            "substitution_cost": 0.0,
            "wastage_cost": 0.0,
            "holding_cost": 0.0,
        }

    # NOTE: New function to support a gymnax env with KPIs using a gymnax fitness object instead of a rollout manager
    # Key difference is that we accumulate KPIs as we go along instead of at the end
    # This should simplif things/make consistent with how we deal with the MARL env
    def calculate_kpis(
        self, cum_info: Dict[str, Union[chex.Array, float]]
    ) -> Dict[str, Union[chex.Array, float]]:
        """Calculate KPIs based on the info recorded by the replenishment agent, with id 0"""
        return {
            "mean_demand_by_pt_blood_group": cum_info["demand"]
            / cum_info["day_counter"],
            "mean_total_demand": cum_info["demand"].sum() / cum_info["day_counter"],
            "mean_order_by_product": cum_info["orders"] / cum_info["day_counter"],
            "service_level_%_by_pt_blood_group": (
                (cum_info["demand"] - cum_info["shortages"]) * 100
            )
            / cum_info["demand"],
            "expiries_%_by_product": ((cum_info["expiries"]) * 100)
            / cum_info["orders"],
            "mean_holding_by_product": cum_info["holding"] / cum_info["day_counter"],
            "mean_age_at_transfusion_by_pt_blood_group": self._calculate_mean_age_at_transfusion_by_pt_blood_group(
                cum_info
            ),
            "exact_match_%_by_pt_blood_group": self._calculate_exact_match_pc_by_pt_blood_group(
                cum_info
            ),
            "mean_total_order": jnp.sum(cum_info["orders"]) / cum_info["day_counter"],
            "service_level_%": (
                jnp.sum(cum_info["demand"] - cum_info["shortages"]) * 100
            )
            / jnp.sum(cum_info["demand"]),
            "unmet_demand_units": jnp.sum(cum_info["shortages"]),
            "expiries_%": (jnp.sum(cum_info["expiries"]) * 100)
            / jnp.sum(cum_info["orders"]),
            "expired_units": jnp.sum(cum_info["expiries"]),
            "mean_holding": jnp.sum(cum_info["holding"]) / cum_info["day_counter"],
            "exact_match_%": self._calculate_exact_match_pc(cum_info),
            "mean_age_at_transfusion": self._calculate_mean_age_at_transfusion(
                cum_info
            ),
            "mean_variable_order_cost": cum_info["variable_order_cost"]
            / cum_info["day_counter"],
            "mean_fixed_order_cost": cum_info["fixed_order_cost"]
            / cum_info["day_counter"],
            "mean_shortage_cost": cum_info["shortage_cost"] / cum_info["day_counter"],
            "mean_substitution_cost": cum_info["substitution_cost"]
            / cum_info["day_counter"],
            "mean_wastage_cost": cum_info["wastage_cost"] / cum_info["day_counter"],
            "mean_holding_cost": cum_info["holding_cost"] / cum_info["day_counter"],
            "all_allocations": cum_info["allocations"].sum(axis=-1),
        }

    @classmethod
    def calculate_target_kpi_penalty(
        cls, kpis: Dict[str, Union[chex.Array, float]], params
    ):
        # TODO Might want to do some rounding here/use jnp.close etc when aiming for
        # 100% service level or 0% expriries for example to avoid issues with floating
        # point precision
        expiry_penalty = (
            jnp.where(kpis["expiries_%"] > params.max_expiry_pc_target, 1, 0)
            * -params.target_kpi_breach_penalty
        )

        service_level_penalty = (
            jnp.where(
                kpis["service_level_%"] < params.min_service_level_pc_target, 1, 0
            )
            * -params.target_kpi_breach_penalty
        )

        matching_penalty = jnp.where(
            kpis["exact_match_%"] < params.min_exact_match_pc_target, 1, 0
        )

        return expiry_penalty + service_level_penalty + matching_penalty

    def _calculate_exact_match_pc_by_pt_blood_group(
        self, cum_info: Dict[str, Union[chex.Array, float]]
    ) -> chex.Array:
        exact_matches_by_request_type = cum_info["allocations"].sum(axis=-1)[
            jnp.arange(cum_info["allocations"].shape[0]),
            jnp.arange(cum_info["allocations"].shape[0]),
        ]
        total_allocated_by_request_type = cum_info["allocations"].sum(axis=(-2, -1))
        return (exact_matches_by_request_type * 100) / total_allocated_by_request_type

    def _calculate_exact_match_pc(
        self, cum_info: Dict[str, Union[chex.Array, float]]
    ) -> chex.Array:
        exact_matches = jnp.trace(cum_info["allocations"].sum(axis=-1))
        total_allocated = jnp.sum(cum_info["allocations"])
        return (exact_matches * 100) / total_allocated

    def _calculate_mean_age_at_transfusion_by_pt_blood_group(
        self, cum_info: Dict[str, Union[chex.Array, float]]
    ) -> chex.Array:
        ages = jnp.arange(cum_info["allocations"].shape[2])
        age_weighted_allocations = cum_info["allocations"] * ages[None, None, :]
        total_age_per_request_type = age_weighted_allocations.sum(axis=(-2, -1))
        total_allocated_per_request_type = jnp.sum(
            cum_info["allocations"], axis=(-2, -1)
        )
        return total_age_per_request_type / total_allocated_per_request_type

    def _calculate_mean_age_at_transfusion(
        self, cum_info: Dict[str, Union[chex.Array, float]]
    ) -> chex.Array:
        ages = jnp.arange(cum_info["allocations"].shape[2])
        age_weighted_allocations = cum_info["allocations"] * ages[None, None, :]
        return jnp.sum(age_weighted_allocations) / jnp.sum(cum_info["allocations"])

    def end_of_warmup_reset(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ):
        """Run at end of warmup period to partially reset State"""
        _, state_reset = self.reset(key, params)
        # We want to keep the stock on hand and in transit, but reset everything else
        return state_reset.replace(
            stock=state.stock,
            in_transit=state.in_transit,
            issue_policy_params=state.issue_policy_params,
        )
