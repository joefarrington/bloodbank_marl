# A single agent environment in Gymnax, with the functionality to add in a issuing policy as an argument
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
import numpy as np
from bloodbank_marl.scenarios.simple_two_product.gymnax_env import (
    EnvState,
    EnvParams,
    EnvObs,
    DemandInfo,
    IssueObs,
    SimpleTwoProductPerishableGymnax,
)


class SimpleTwoProductPerishableLimitDemandGymnax(SimpleTwoProductPerishableGymnax):
    """Jittable abstract base class for all gymnax Environments."""

    def __init__(
        self,
        n_products: int = 2,
        max_useful_life: int = 2,
        lead_time: int = 1,
        max_order_quantity: int = 10,  # Applies to any individual product, rather than in total
        max_demand: int = 100,  # TODO: Check older work
        issuing_policy=None,  # TODO: For now, set as None
    ):
        super().__init__()
        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantities = jnp.array([max_order_quantity] * self.n_products)
        self.max_demand = max_demand
        self._issuing_policy = issuing_policy

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
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
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
        request_types_raw = distrax.Categorical(
            probs=params.product_probabilities
        ).sample(seed=type_key, sample_shape=(self.max_demand,))
        request_types = jnp.where(cum_time < 0.75, request_types_raw, 1)
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

    @property
    def name(self) -> str:
        """Environment name."""
        return "SimpleTwoProductPerishableLimitDemandGymnax"
