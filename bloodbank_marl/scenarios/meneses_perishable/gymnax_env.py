# A single agent environment in Hymnax, with the functionality to add in a issuing policy as an argument
# Not true multiagent, because we don't get states for other agents, but we can use this to test the issuing policy and
# the rep policy at the same time.
# Currently a lot quicker than the multiagent env, so much better for testing "fixed" issuing policies like exact match
# and fixed priority.

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

# TODO: Fill out missing methods (e.g. for action/obs/state spaces etc) if we end up using this for final results

# TODO: Decide what to do about EnvObs and policies - can write specific policies for the single
# agent env, do what we currently do here, or rethink the use of them if it is what is slowing down
# the multiagent env.

# TODO THink about best way to have default issuing policy, and use config to specify another
# exact_match = FixedPolicy(exact_match_policy, None, {})

# TODO: Decide where to specify n_products and how to get it into default params
n_products = 8
C = 1e10  # invalid substitution cost


@struct.dataclass
class EnvState:
    stock: chex.Array
    in_transit: chex.Array
    step: int  # For the purposes of this env, a step is a day


@struct.dataclass
class EnvParams:
    poisson_demand_mean: float
    product_probabilities: chex.Array
    age_on_arrival_distribution_probs: chex.Array
    fixed_order_cost: float
    variable_order_costs: chex.Array
    shortage_costs: chex.Array
    wastage_costs: chex.Array
    holding_costs: chex.Array
    # For now, we assume that subsitution cost increases by 1/8
    # TODO: Check if this is what they meant (or whether, for example, if only one possible sub
    # then it is the wost and so should be 7/8)
    substitution_costs: chex.Array
    # TODO: Add in functions to apply penalty for breaching targets
    max_expiry_pc_target: float
    min_service_level_pc_target: float
    min_exact_match_pc_target: float
    target_kpi_breach_penalty: float
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        cls,
        poisson_demand_mean: float = 49.8,
        product_probabilities: List[float] = [
            0.08614457,
            0.36024097,
            0.08192771,
            0.36485943,
            0.0126506,
            0.0684739,
            0.00522088,
            0.02048193,
        ],
        age_on_arrival_distribution_probs: List[float] = [1] + [0] * (35 - 1),
        fixed_order_cost: float = 0,
        variable_order_costs: List[float] = [160] * n_products,
        shortage_costs: List[float] = [1340] * n_products,
        wastage_costs: List[float] = [130] * n_products,
        holding_costs: List[float] = [1.1] * n_products,
        substitution_cost_ratios: List[List[float]] = [
            # Unit O-, O+, A-, A+, B-, B+, AB-, AB+
            [0, C, C, C, C, C, C, C],  # O- pt
            [
                1 / 8,
                0,
                C,
                C,
                C,
                C,
                C,
                C,
            ],  # O+ pt
            [
                1 / 8,
                C,
                0,
                C,
                C,
                C,
                C,
                C,
            ],  # A- pt
            [3 / 8, 2 / 8, 1 / 8, 0, C, C, C, C],  # A+ pt
            [
                1 / 8,
                C,
                C,
                C,
                0,
                C,
                C,
                C,
            ],  # B- pt
            [3 / 8, 2 / 8, C, C, 1 / 8, 0, C, C],  # B+ pt
            [3 / 8, C, 2 / 8, C, 1 / 8, C, 0, C],  # AB- pt
            [7 / 8, 6 / 8, 5 / 8, 4 / 8, 3 / 8, 2 / 8, 1 / 8, 0],  # AB+ pt
        ],
        max_substitution_cost: float = 1340,
        max_expiry_pc_target: float = 100.0,  # Effectively no limit by default
        min_service_level_pc_target: float = 0.0,  # Effectively no limit by default
        min_exact_match_pc_target: float = 0.0,  # Effectively no limit by default
        target_kpi_breach_penalty: float = 1e10,
        max_steps_in_episode: int = 1e10,
        gamma: float = 1.0,
    ):
        return cls(
            poisson_demand_mean,
            jnp.array(product_probabilities),
            jnp.array(age_on_arrival_distribution_probs),
            fixed_order_cost,
            jnp.array(variable_order_costs),
            jnp.array(shortage_costs),
            jnp.array(wastage_costs),
            jnp.array(holding_costs),
            jnp.array(substitution_cost_ratios) * max_substitution_cost,
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

    @property
    def obs(self):
        return jnp.hstack(
            [
                self.in_transit.reshape(-1),
                self.stock.reshape(-1),
            ]
        )


@struct.dataclass
class DemandInfo:
    total_demand: int
    remaining_demand: int
    shortages: int
    allocations: int
    remaining_stock: chex.Array
    request_type_samples: chex.Array
    key: jax.random.PRNGKey


@struct.dataclass
class IssueObs:
    stock: chex.Array
    request_type: int


class MenesesPerishableGymnax(environment.Environment):
    """Jittable abstract base class for all gymnax Environments."""

    def __init__(
        self,
        n_products: int = 8,
        max_useful_life: int = 35,
        lead_time: int = 1,
        max_order_quantities: list = [500] * 8,
        max_demand: int = 500,
        issuing_policy=None,  # TODO: For now, set as None
    ):
        super().__init__()
        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantities = jnp.array(max_order_quantities)
        self.max_demand = max_demand
        self._issuing_policy = issuing_policy

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
        demand_key, type_key, arrival_key = jax.random.split(key, 3)

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
        remaining_demand = jnp.clip(
            jax.random.poisson(demand_key, params.poisson_demand_mean),
            0,
            self.max_demand,
        )
        info["total_demand"] = remaining_demand

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
            request_types,
            arrival_key,
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
        info["demand_by_pt_blood_group"] = shortages + allocations.sum(axis=(-2, -1))

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
            stock=stock, in_transit=in_transit, step=state.step + 1
        )  # TODO Add in the other elements
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state).obs),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        # When lead time is 0, still have one slot to put order into and then immedifately remove from for simplicity
        # when using receive order functions
        state = EnvState(
            stock=jnp.zeros((self.n_products, self.max_useful_life)),
            in_transit=jnp.zeros((self.n_products, max(self.lead_time, 1))),
            step=0,
        )
        return self.get_obs(state).obs, state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        # If lead time is 0 or 1, nothing to include in obs
        return EnvObs(stock=state.stock, in_transit=state.in_transit[:, 1:])

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
        return "MenesesPerishableGymnax"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        raise NotImplementedError

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
            stock=demand_info.remaining_stock, request_type=requested_product_idx
        )
        issue_action = self._issuing_policy.apply(None, issuing_obs, issue_key)
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
            demand_info.request_type_samples,
            key,
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

    @classmethod
    def calculate_kpis(cls, info: Dict) -> Dict[str, float]:
        """Calculate KPIs for each rollout, using the output of a rollout from RolloutWrapper"""
        mean_demand_by_pt_blood_group = info["demand_by_pt_blood_group"].mean(axis=-2)
        mean_order_by_product = info["orders"].mean(axis=-2)
        service_level_pc_by_pt_blood_group = (
            (
                info["demand_by_pt_blood_group"].sum(axis=-2)
                - info["shortages"].sum(axis=-2)
            )
            * 100
        ) / info["demand_by_pt_blood_group"].sum(axis=-2)
        expiries_pc_by_product = (info["expiries"].sum(axis=-2) * 100) / info[
            "orders"
        ].sum(axis=-2)
        mean_holding_by_product = info["holding"].mean(axis=-2)
        mean_age_at_transfusion_by_pt_blood_group = (
            cls._calculate_mean_age_at_transfusion_by_pt_blood_group(info)
        )
        exact_match_pc_by_pt_blood_group = (
            cls._calculate_exact_match_pc_by_pt_blood_group(info)
        )
        mean_total_order = info["orders"].sum(axis=-1).mean(axis=-1)
        service_level_pc = (
            (info["total_demand"].sum(axis=(-1)) - info["shortages"].sum(axis=(-2, -1)))
            * 100
        ) / info["total_demand"].sum(axis=(-1))
        expiries_pc = (info["expiries"].sum(axis=(-2, -1)) * 100) / info["orders"].sum(
            axis=(-2, -1)
        )
        mean_holding = info["holding"].sum(axis=-1).mean(axis=-1)
        exact_match_pc = cls._calculate_exact_match_pc(info)
        mean_age_at_transfusion = cls._calculate_mean_age_at_transfusion(info)
        unmet_demand_units = info["shortages"].sum(axis=(-2, -1))
        expired_units = info["expiries"].sum(axis=(-2, -1))

        return {
            "mean_demand_by_pt_blood_group": mean_demand_by_pt_blood_group,
            "mean_order_by_product": mean_order_by_product,
            "service_level_%_by_pt_blood_group": service_level_pc_by_pt_blood_group,
            "expiries_%_by_product": expiries_pc_by_product,
            "mean_holding_by_product": mean_holding_by_product,
            "mean_age_at_transfusion_by_pt_blood_group": mean_age_at_transfusion_by_pt_blood_group,
            "exact_match_%_by_pt_blood_group": exact_match_pc_by_pt_blood_group,
            "mean_total_order": mean_total_order,
            "service_level_%": service_level_pc,
            "expiries_%": expiries_pc,
            "mean_holding": mean_holding,
            "exact_match_%": exact_match_pc,
            "mean_age_at_transfusion": mean_age_at_transfusion,
            "unmet_demand_units": unmet_demand_units,
            "expired_units": expired_units,
        }

    @classmethod
    def _calculate_exact_match_pc_by_pt_blood_group(cls, info: Dict) -> chex.Array:
        """Calculate the exact match percentage by product type and blood group"""
        n_groups = info["allocations"].shape[-2]
        exact_matches_by_pt_blood_group = info["allocations"].sum(axis=(-4, -1))[
            jnp.arange(n_groups), jnp.arange(n_groups)
        ]
        total_allocated_by_pt_blood_group = info["allocations"].sum(axis=(-4, -2, -1))
        return (
            exact_matches_by_pt_blood_group * 100
        ) / total_allocated_by_pt_blood_group

    @classmethod
    def _calculate_exact_match_pc(cls, info: Dict) -> float:
        """Calculate the exact match percentage"""
        exact_matches = jnp.trace(
            info["allocations"].sum(axis=(-4, -1)),
            axis1=-2,
            axis2=-1,
        )
        total_allocated = jnp.sum(info["allocations"].sum(axis=(-4, -3, -2, -1)))
        return (exact_matches * 100) / total_allocated

    @classmethod
    def _calculate_mean_age_at_transfusion_by_pt_blood_group(
        cls, info: Dict
    ) -> chex.Array:
        """Calculate the mean age at transfusion by patient blood group (i.e. irrespective of what they were allocated)"""
        # TODO: We might also want to do this by product type
        ages = jnp.arange(info["allocations"].shape[-1])
        age_weighted_allocations = (
            info["allocations"].sum(axis=-4) * ages[None, None, :]
        )
        total_age_per_request_type = age_weighted_allocations.sum(axis=(-2, -1))
        total_allocated_per_request_type = info["allocations"].sum(axis=(-4, -2, -1))
        return total_age_per_request_type / total_allocated_per_request_type

    @classmethod
    def _calculate_mean_age_at_transfusion(cls, info: Dict) -> float:
        """Calculate the mean age at transfusion"""
        ages = jnp.arange(info["allocations"].shape[-1])
        age_weighted_allocations = info["allocations"].sum(axis=(-4, -3, -2)) * ages
        return jnp.sum(age_weighted_allocations) / jnp.sum(info["allocations"])

    @classmethod
    def calculate_target_kpi_penalty(self, kpis: Dict, params: EnvParams) -> float:
        """Calculate the penalty for breaching the target KPIs"""
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

        exact_match_penalty = (
            jnp.where(kpis["exact_match_%"] < params.min_exact_match_pc_target, 1, 0)
            * -params.target_kpi_breach_penalty
        )

        return expiry_penalty + service_level_penalty + exact_match_penalty
