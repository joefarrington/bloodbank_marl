import jax
import chex
from typing import Tuple, Union, Optional, Dict, List
from flax import struct
import jax.numpy as jnp
from gymnax.environments import spaces
import numpy as np
import distrax
from bloodbank_marl.scenarios.simple_two_product.jax_env import (
    EnvParams,
    EnvState,
    EnvInfo,
    EnvObs,
    SimpleTwoProductPerishableEnv,
)
from jax import lax

jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32

# NOTE: Differences to single agent version
# 1) state need to include full max_useful_life, because for issuing actions, all will be available
# and we need them to have the same dimensions.


class SimpleTwoProductPerishableLimitDemandEnv(SimpleTwoProductPerishableEnv):
    def __init__(
        self,
        agent_names=["replenishment", "issuing"],
        n_products: int = 2,
        max_useful_life: int = 2,
        lead_time: int = 1,
        max_order_quantity: int = 10,
        max_demand=100,
    ):
        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantities = jnp.array([max_order_quantity] * self.n_products)
        self.max_demand = max_demand

        self.possible_agents = agent_names
        self.agent_ids = {agent_name: i for i, agent_name in enumerate(agent_names)}
        self.num_agents = len(agent_names)

    @property
    def name(self) -> str:
        """Environment name."""
        return "SimpleTwoProductPerishableLimitDemand"

    def _replenishment_step(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> EnvState:
        """Replenishment action step."""
        interval_key, type_key, arrival_key = jax.random.split(key, 3)

        stock, in_transit, infos, cumulative_rewards = (
            state.stock,
            state.in_transit,
            state.infos,
            state.cumulative_rewards,
        )
        orders = jnp.clip(action, a_min=0, a_max=self.max_order_quantities)
        order_placed = jax.lax.cond(jnp.sum(orders) > 0, lambda: 1, lambda: 0)

        # Place the order
        variable_order_cost = jnp.dot(orders, -params.variable_order_costs)
        fixed_order_cost = order_placed * -params.fixed_order_costs
        cumulative_rewards = cumulative_rewards + variable_order_cost + fixed_order_cost

        in_transit = in_transit.at[0 : self.n_products, 0].set(orders)

        # If L==0, stock recieves immediately, otherwise no change
        stock, in_transit = jax.lax.cond(
            self.lead_time == 0,
            self._receive_order,
            lambda arrival_key, stock, in_transit, params: (stock, in_transit),
            arrival_key,
            stock,
            in_transit,
            params,
        )
        # Sample the demand for the coming day
        # NOTE: In prev version of RS - we make decision on, say Sunday eve then arrives first thing Monday morning
        # So demand to same for the step when weekday in the state is Sunday is Monday's demand
        # Here, for simplicity, we just assume observation point is first thing in the morning,
        # So if the weekday in the state is Monday (0) then we sample demand for Monday
        # Basically, we update weekday just before a replenishment step instead of at the very beginned of the step
        request_intervals = distrax.Gamma(
            concentration=1, rate=params.poisson_demand_mean
        ).sample(seed=interval_key, sample_shape=(self.max_demand,))

        # NOTE: Simple version of this env for now, after half way point of the day only demand from Product B
        # So, there may be a benefit to the issuing policy being time aware.

        cum_time = jnp.cumsum(request_intervals)
        request_types_raw = distrax.Categorical(
            probs=params.product_probabilities
        ).sample(seed=type_key, sample_shape=(self.max_demand,))
        request_types = jnp.where(cum_time < 0.75, request_types_raw, 1)

        # Update infos
        infos = infos.replace(
            orders=infos.orders.at[:].add(orders),
            order_placed=infos.order_placed.at[:].add(order_placed),
        )

        state = state.replace(
            stock=stock,
            in_transit=in_transit,
            infos=infos,
            request_intervals=request_intervals,
            request_types=request_types,
            request_idx=0,
            cumulative_rewards=cumulative_rewards,
        )

        return state
