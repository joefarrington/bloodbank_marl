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
from bloodbank_marl.scenarios.simple_two_product.gymnax_env_try_issue_too import (
    SimpleTwoProductPerishableIncIssueGymnax,
)

# TODO: Fill out missing methods (e.g. for action/obs/state spaces etc) if we end up using this for final results

# TODO: Decide where to specify n_products and how to get it into default params

n_products = 8
M = 1e10  # invalid substitution cost
max_useful_life = 3

# Base on Yu Suen et al (2023)
substitution_cost_ratios = [
    # Unit O-, O+, A-, A+, B-, B+, AB-, AB+
    [
        0,
        M,
        2,
        M,
        1,
        M,
        M,
        M,
    ],  # O- patient
    [1, 0, 5, 4, 3, 2, M, M],  # O+ patient
    [
        3,
        M,
        0,
        4,
        2,
        M,
        1,
        5,
    ],  # A- patient
    [M, M, 1, 0, 5, 4, 3, 2],  # A+ patient
    [
        3,
        M,
        2,
        M,
        0,
        4,
        1,
        5,
    ],  # B- pt
    [M, M, 5, 4, 1, 0, 3, 2],  # B+ patient
    [3, M, 1, 5, 2, M, 0, 4],  # AB- patient
    [M, M, 3, 2, 5, 4, 1, 0],  # AB+ patient
]

# Preferences for issuing policy based on the above
# [0, 4, 2, -1, -1, -1, -1, -1] # O- patient
# [1, 0, 5, 4, 3, 2, -1, -1] # O+ patient
# [2, 6, 4, 0, 3, 7, -1, -1] # A- patient
# [3, 2, 7, 6, 5, 4, -1, -1] # A+ patient
# [4, 6, 2, 0, 5, 7, -1, -1] # B- patient
# [5,4, 7, 6, 3, 2, -1, -1] #B+ patient
# [6, 2, 4, 0, 7, 3, -1, -1] # AB- patient
# [7, 6, 3, 2, 5, 4, -1, -1] # AB+ patient

# These are from Ensafian et al (2017) - and similar to those in Meneses
product_probabilities = [
    0.07,  # O-
    0.38,  # O+
    0.06,  # A-
    0.34,  # A+
    0.02,  # B-
    0.09,  # B+
    0.01,  # AB-
    0.03,  # AB+
]

action_mask_per_request_type = jnp.where(
    jnp.array(substitution_cost_ratios) > n_products, 0, 1
)


@struct.dataclass
class EnvState:
    stock: chex.Array
    in_transit: chex.Array
    weekday: int
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
    initial_weekday: int
    max_wastage_pc_target: float
    min_service_level_pc_target: float
    min_exact_match_pc_target: float
    target_kpi_breach_penalty: float
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        cls,
        poisson_demand_mean: List[float] = [
            37.5,
            37.3,
            39.2,
            37.8,
            40.5,
            27.2,
            28.4,
        ],
        product_probabilities: List[float] = product_probabilities,
        age_on_arrival_distribution_probs: List[float] = [1]
        + [0] * (max_useful_life - 1),
        fixed_order_cost: float = 0,
        variable_order_costs: List[float] = [650.0] * n_products,
        shortage_costs: List[float] = [3250.0] * n_products,
        wastage_costs: List[float] = [650] * n_products,
        holding_costs: List[float] = [130] * n_products,
        substitution_cost_ratios: List[List[float]] = substitution_cost_ratios,
        max_substitution_cost: float = 3250.0,
        action_mask_per_request_type: List[int] = action_mask_per_request_type,
        initial_weekday: int = -1,
        max_wastage_pc_target: float = 100.0,  # Effectively no limit by default
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
            jnp.array(fixed_order_cost),
            jnp.array(variable_order_costs),
            jnp.array(shortage_costs),
            jnp.array(wastage_costs),
            jnp.array(holding_costs),
            jnp.array(substitution_cost_ratios) * (max_substitution_cost / n_products),
            jnp.array(action_mask_per_request_type),
            initial_weekday,
            max_wastage_pc_target,
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
    weekday: chex.Array
    action_mask: chex.Array

    @property
    def obs(self):
        batch_dims = self.in_transit.shape[:-2]
        return jnp.hstack(
            [
                self.one_hot_day_of_week().reshape(batch_dims + (-1,)),
                self.in_transit.reshape(batch_dims + (-1,)),
                self.stock.reshape(batch_dims + (-1,)),
            ]
        )

    @property
    def obs_total_per_product(self):
        batch_dims = self.in_transit.shape[:-2]
        return self.total_per_product().reshape(batch_dims + (-1,))

    @property
    def obs_total_per_product_and_weekday(self):
        batch_dims = self.in_transit.shape[:-2]
        return jnp.hstack(
            [
                self.one_hot_day_of_week().reshape(batch_dims + (-1,)),
                self.total_per_product().reshape(batch_dims + (-1,)),
            ]
        )

    def one_hot_day_of_week(self):
        return jax.nn.one_hot(self.weekday, 7)

    def total_per_product(self):
        return self.stock.sum(axis=-1) + self.in_transit.sum(axis=-1)


@struct.dataclass
class DemandInfo:
    total_demand: int
    remaining_demand: int
    shortages: int
    allocations: int
    remaining_stock: chex.Array
    in_transit: chex.Array
    weekday: int
    request_type_samples: chex.Array
    request_time_samples: chex.Array
    key: jax.random.PRNGKey
    issue_policy_params: Optional[Dict]
    action_mask_per_request_type: chex.Array


@struct.dataclass
class IssueObs:
    stock: chex.Array
    in_transit: chex.Array
    request_time: chex.Array
    request_type: chex.Array
    weekday: chex.Array
    action_mask: chex.Array

    @property
    def obs(self):
        batch_dims = self.in_transit.shape[:-2]
        return jnp.hstack(
            [
                self.one_hot_day_of_week().reshape(batch_dims + (-1,)),
                self.request_time.reshape(batch_dims + (-1,)),
                self.one_hot_request_type.reshape(batch_dims + (-1,)),
                self.in_transit.reshape(batch_dims + (-1,)),
                self.stock.reshape(batch_dims + (-1,)),
            ]
        )

    def one_hot_day_of_week(self):
        return jax.nn.one_hot(self.weekday, 7)

    def one_hot_request_type(self):
        return jax.nn.one_hot(self.request_type, n_products)


class RSPerishableIncIssueGymnax(SimpleTwoProductPerishableIncIssueGymnax):
    """Jittable abstract base class for all gymnax Environments."""

    def __init__(
        self,
        n_products: int = n_products,
        max_useful_life: int = 3,
        lead_time: int = 0,
        max_order_quantity: int = 50,  # Applies to any individual product, rather than in total
        max_demand: int = 100,
    ):
        super().__init__()
        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantities = jnp.array([max_order_quantity] * self.n_products)
        self.max_demand = max_demand

    def default_issue_obs(self, params: EnvParams) -> IssueObs:
        return IssueObs(
            weekday=jnp.array(0),
            stock=jnp.zeros((self.n_products, self.max_useful_life)),
            in_transit=jnp.zeros((self.n_products, self.lead_time)),
            request_time=jnp.array(0.0),
            request_type=jnp.array(0),
            action_mask=jnp.ones((self.n_products,)),
        )

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

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
            concentration=1, rate=params.poisson_demand_mean[state.weekday]
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
            state.weekday,
            request_types,
            cum_time,
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
            weekday=(state.weekday + 1) % 7,
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
        # If initial weekday == -1, randomly sample it
        key, weekday_key = jax.random.split(key)
        initial_weekday = jax.lax.select(
            params.initial_weekday >= 0,
            params.initial_weekday % 7,
            jax.random.randint(
                weekday_key, shape=(), minval=0, maxval=7, dtype=jnp.int32
            ),
        )

        # When lead time is 0, still have one slot to put order into and then immediately remove from for simplicity
        # when using receive order functions
        state = EnvState(
            stock=jnp.zeros((self.n_products, self.max_useful_life)),
            # The in_transit element of state should have at least one element, even if L=0
            in_transit=jnp.zeros((self.n_products, max(self.lead_time, 1))),
            weekday=initial_weekday,
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""

        # Simple action masking, for now can always order each product
        return EnvObs(
            stock=state.stock,
            # If lead time is 0 or 1, nothing to include in replenishment obs
            in_transit=state.in_transit[:, 1 : self.lead_time],
            weekday=state.weekday,
            action_mask=jnp.ones((self.n_products,)),
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "RSPerishableIncIssueGymnax"

    def action_space(self, params: EnvParams):
        """Action space of the environment."""
        raise NotImplementedError

    def observation_space(self, params: EnvParams):
        """Observation space of the environment."""
        raise NotImplementedError

    def state_space(self, params: EnvParams):
        """State space of the environment."""
        raise NotImplementedError

    def _fill_one_request(self, demand_info: DemandInfo) -> DemandInfo:
        idx = demand_info.total_demand - demand_info.remaining_demand
        key, issue_key = jax.random.split(demand_info.key)
        remaining_demand = demand_info.remaining_demand - 1
        requested_product_idx = demand_info.request_type_samples[idx]
        request_time = demand_info.request_time_samples[idx]

        # Identify the product type to be issued
        issuing_obs = IssueObs(
            stock=demand_info.remaining_stock,
            # Additional element of lead time compared to replenishment obs if L>0
            # because there is stock in transit during the day we may need to account for
            in_transit=demand_info.in_transit[:, 0 : self.lead_time],
            weekday=demand_info.weekday,
            request_time=request_time,
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
            demand_info.weekday,
            demand_info.request_type_samples,
            demand_info.request_time_samples,
            key,
            demand_info.issue_policy_params,
            demand_info.action_mask_per_request_type,
        )

    def end_of_warmup_reset(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ):
        """Run at end of warmup period to partially reset State"""
        _, state_reset = self.reset(key, params)
        return state_reset.replace(
            stock=state.stock,
            in_transit=state.in_transit,
            weekday=state.weekday,
            issue_policy_params=state.issue_policy_params,
        )
