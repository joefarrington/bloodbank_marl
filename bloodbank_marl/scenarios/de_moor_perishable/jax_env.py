import jax
import chex
from typing import Tuple, Union, Optional, Dict
from functools import partial
from flax import struct
import jax.numpy as jnp
from gymnax.environments import spaces
import numpy as np
import evosax
import gymnax
from flax import linen as nn
from evosax import OpenES, PGPE, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
from evosax.problems import GymnaxFitness
import distrax
import functools
from bloodbank_marl.environments.environment import (
    MarlEnvironment,
    EnvParams,
    EnvState,
    EnvInfo,
    EnvObs,
)
from jax import lax

jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32

# TODO: Create base class for multiagent JAX env and inherit from that


# TODO: Consider if gamma goes here
@struct.dataclass
class EnvParams:
    max_demand: int
    demand_gamma_alpha: float
    demand_gamma_beta: float
    variable_order_cost: float
    shortage_cost: float
    wastage_cost: float
    holding_cost: float
    max_days_in_episode: int
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        # Default env params are for m=2, experiment 1
        cls,
        max_demand: int = 100,
        demand_gamma_mean: float = 4.0,
        demand_gamma_cov: float = 0.5,
        variable_order_cost: float = 3.0,
        shortage_cost: float = 5.0,
        wastage_cost: float = 7.0,
        holding_cost: float = 1.0,
        max_days_in_episode: int = 365,
        max_steps_in_episode: int = 1e6,
        gamma: float = 0.99,
    ):
        demand_gamma_alpha = 1 / (demand_gamma_cov**2)
        demand_gamma_beta = 1 / (demand_gamma_mean * demand_gamma_cov**2)
        return cls(
            max_demand,
            demand_gamma_alpha,
            demand_gamma_beta,
            variable_order_cost,
            shortage_cost,
            wastage_cost,
            holding_cost,
            max_days_in_episode,
            max_steps_in_episode,
            gamma,
        )


@struct.dataclass
class EnvInfo:
    demand: chex.Array
    shortages: chex.Array
    expiries: chex.Array
    holding: chex.Array
    orders: chex.Array
    order_placed: chex.Array
    day_counter: chex.Array

    @classmethod
    def create_empty_infos(cls, num_agents: int):
        return cls(
            demand=jnp.zeros(num_agents, dtype=jnp.int32),
            shortages=jnp.zeros(num_agents, dtype=jnp.int32),
            expiries=jnp.zeros(num_agents, dtype=jnp.int32),
            holding=jnp.zeros(num_agents, dtype=jnp.int32),
            orders=jnp.zeros(num_agents, dtype=jnp.int32),
            order_placed=jnp.zeros(num_agents, dtype=jnp.int32),
            day_counter=jnp.zeros(num_agents, dtype=jnp.int32),
        )

    def reset_infos_one_agent(self, agent_id: int):
        """Reset the infos for a single agent"""
        return self.replace(
            demand=self.demand.at[agent_id].set(0),
            shortages=self.shortages.at[agent_id].set(0),
            expiries=self.expiries.at[agent_id].set(0),
            holding=self.holding.at[agent_id].set(0),
            orders=self.orders.at[agent_id].set(0),
            order_placed=self.order_placed.at[agent_id].set(0),
            day_counter=self.day_counter.at[agent_id].set(0),
        )

    def accumulate_infos_one_agent(self, agent_id: int, step_info: EnvInfo):
        return self.replace(
            demand=self.demand.at[agent_id].add(step_info.demand),
            shortages=self.shortages.at[agent_id].add(step_info.shortages),
            expiries=self.expiries.at[agent_id].add(step_info.expiries),
            holding=self.holding.at[agent_id].add(step_info.holding),
            orders=self.orders.at[agent_id].add(step_info.orders),
            order_placed=self.order_placed.at[agent_id].add(step_info.order_placed),
            day_counter=self.day_counter.at[agent_id].add(step_info.day_counter),
        )

    def calculate_kpis(self) -> Dict[str, int]:
        """Calculate KPIs from cumulative info for the replenishment agent"""

        return {
            "service_level_%": (self.demand[0] - self.shortages[0]) / self.demand[0],
            "wastage_%": self.expiries[0] / self.orders[0],
            "holding": self.holding[0] / self.day_counter[0],
        }

    @classmethod
    def calculate_target_kpi_penalty(
        cls, kpis: Dict[str, Union[chex.Array, float]], params: EnvParams
    ):
        return 0.0


@struct.dataclass
class EnvState:
    stock: chex.Array
    in_transit: chex.Array
    remaining_demand: int
    agent_id: int  # Agent to act
    cumulative_rewards: chex.Array
    infos: EnvInfo
    truncations: chex.Array
    terminations: chex.Array
    live_agents: chex.Array
    day: int  # Measures days
    step: int  # Measure total steps, rep and alloc combined


@struct.dataclass
class EnvObs:
    agent_id: int  # Following Tianhou;
    in_transit: chex.Array
    stock: chex.Array
    action_mask: chex.Array

    @property
    def obs(self):
        return jnp.hstack([self.in_transit, self.stock])

    @classmethod
    def create_empty_obs(
        cls,
        env_kwargs={"max_useful_life": 2, "lead_time": 1, "max_order_quantity": 10},
        n_steps=1,
    ):
        # For replenishment, action size is max_order_quantity + 1, for issuing it's max_useful_life + 1
        # If we want to use action masking, which we do for issuing at least, we need the action mask
        # to be the same dimensions in the observation for both agents in order to work with JIT.
        # We need a consistent action size, so pick the largest and use masking to ensure only
        # valid actions are taken
        action_dim = (
            jnp.maximum(env_kwargs["max_useful_life"], env_kwargs["max_order_quantity"])
            + 1
        )
        return cls(
            agent_id=jnp.zeros(n_steps, dtype=jnp.int32).squeeze(),
            in_transit=jnp.zeros(
                (n_steps, env_kwargs["lead_time"] - 1), dtype=jnp.int32
            ).squeeze(),
            stock=jnp.zeros(
                (n_steps, env_kwargs["max_useful_life"]), dtype=jnp.int32
            ).squeeze(),
            action_mask=jnp.zeros((n_steps, action_dim), dtype=jnp.int32).squeeze(),
        )


class DeMoorPerishableMAJAX(MarlEnvironment):
    """Jittable abstract base class for all gymnax-inspired Multi-Agent Environments."""

    def __init__(
        self,
        agent_names=["replenishment", "issuing"],
        max_useful_life: int = 2,
        lead_time: int = 1,
        max_order_quantity: int = 10,
    ):
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantity = max_order_quantity

        self.possible_agents = agent_names
        self.agent_ids = {agent_name: i for i, agent_name in enumerate(agent_names)}
        self.num_agents = len(agent_names)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    @property
    def empty_infos(self) -> EnvInfo:
        return EnvInfo.create_empty_infos(self.num_agents)

    @property
    def empty_obs(self, n_steps=1) -> EnvObs:
        return EnvObs.create_empty_obs(
            env_kwargs={
                "max_useful_life": self.max_useful_life,
                "lead_time": self.lead_time,
                "max_order_quantity": self.max_order_quantity,
            },
            n_steps=n_steps,
        )

    def live_step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[EnvObs, EnvState, float, bool, bool, EnvInfo]:
        # Zero the cumulative reward and reset the info for the agent that just stepped
        infos, cumulative_rewards = state.infos, state.cumulative_rewards
        cumulative_rewards = jnp.where(
            jnp.arange(2) == state.agent_id, 0, state.cumulative_rewards
        )
        infos = infos.reset_infos_one_agent(state.agent_id)
        state = state.replace(infos=infos, cumulative_rewards=cumulative_rewards)

        # TODO: Difference between truncation and termination
        state = jax.lax.switch(
            state.agent_id,
            [self._replenishment_step, self._issuing_step],
            key,
            state,
            action,
            params,
        )
        # Select the next agent
        # If no remainine demand, next agent is replenishment (agent_id 0)
        next_agent_id = jax.lax.cond(state.remaining_demand == 0, lambda: 0, lambda: 1)

        # If the next agent is replenishent, we need to age the stock, work out the holding cost, update the time
        state = jax.lax.cond(
            next_agent_id == 0,
            self._age_stock,
            lambda state, params: state,
            state,
            params,
        )
        day_increment = jax.lax.cond(next_agent_id == 0, lambda: 1, lambda: 0)
        state = state.replace(
            infos=state.infos.replace(
                day_counter=infos.day_counter.at[:].add(day_increment)
            )
        )

        # Check for termination - for now we don't make a distinction between truncation and termination
        # And either both or noen of the agents are done
        # TODO: Separate termination for each agent? Not necessary for what we're doing.
        trunc = jax.lax.cond(
            jax.lax.bitwise_or(
                state.day + day_increment >= params.max_days_in_episode,
                state.step + 1 >= params.max_steps_in_episode,
            ),
            lambda _: True,
            lambda _: False,
            None,
        )
        truncations = state.truncations.at[:].set(trunc)
        live = jax.lax.cond(truncations[state.agent_id], lambda: 0, lambda: 1)
        live_agents = state.live_agents.at[state.agent_id].set(live)

        # Update the state
        state = state.replace(
            agent_id=next_agent_id,
            truncations=truncations,
            live_agents=live_agents,
            day=state.day + day_increment,
            step=state.step + 1,
        )
        return (
            lax.stop_gradient(self.get_obs(state, params, next_agent_id)),
            lax.stop_gradient(state),
            state.cumulative_rewards[next_agent_id],
            state.truncations[next_agent_id],
            state.terminations[next_agent_id],
            self.get_info(state, next_agent_id),
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[EnvObs, EnvState]:
        """Environment-specific reset."""
        state = EnvState(
            stock=jnp.zeros(self.max_useful_life, dtype=jnp.int32),
            in_transit=jnp.zeros(self.lead_time, dtype=jnp.int32),
            remaining_demand=0,
            agent_id=0,
            cumulative_rewards=jnp.array([0.0] * len(self.agent_ids)),
            infos=self.empty_infos,
            truncations=jnp.array([False] * len(self.agent_ids)),
            terminations=jnp.array([False] * len(self.agent_ids)),
            live_agents=jnp.array([1] * len(self.agent_ids)),
            day=0,
            step=0,
        )
        return self.get_obs(state, params, state.agent_id), state

    def end_of_warmup_reset(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ):
        """Run at end of warmup period to partially reset State"""
        _, state_reset = self.reset(key, params)
        # We want to keep the stock on hand and in transit, but reset everything else
        return state_reset.replace(stock=state.stock, in_transit=state.in_transit)

    def get_obs(self, state: EnvState, params: EnvParams, agent_id: int) -> EnvObs:
        """Applies observation function to state, in PettinZoo AECEnv the equivalent is .observe()"""
        # TODO: For now, each agent gets the same observation and we'll deal with it at the agent level
        return EnvObs(
            state.agent_id,
            state.in_transit[1 : self.lead_time + 1],
            state.stock,
            self._get_action_mask(state, params, agent_id),
        )

    def _get_action_mask(
        self, state: EnvState, params: EnvParams, agent_id: int
    ) -> chex.Array:
        """Get action mask for agent with id `agent_id`."""
        return jax.lax.switch(
            agent_id,
            [
                lambda x, y: self._get_replenishment_mask(x, y),
                lambda x, y: self._get_issuing_mask(x, y),
            ],
            state,
            params,
        )

    def _get_replenishment_mask(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Get action mask for replenishment agent."""
        return jnp.ones(self.max_order_quantity + 1, dtype=jnp.int32)

    def _get_issuing_mask(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Get action mask for issuing agent."""
        base_mask = jnp.where(state.stock > 0, 1, 0)
        # Issuing nothing (action 0) always allowed, then one action per age if in stock, then pad with zeros
        return jnp.hstack(
            [
                jnp.array([1]),
                base_mask,
                jnp.zeros(
                    self.max_order_quantity - self.max_useful_life, dtype=jnp.int32
                ),
            ]
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""
        return jax.lax.cond(
            jax.lax.bitwise_or(
                state.day >= params.max_days_in_episode,
                state.step >= params.max_steps_in_episode,
            ),
            lambda _: True,
            lambda _: False,
            None,
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "DeMoorPerishable"

    def num_actions(self, agent_id: int) -> int:
        """Number of actions possible in environment for agent with id `agent_id`."""
        # NOTE: We use this to get the output dim of the policy network
        # Therefore, same for both agents, the maximum of the two
        return self.action_space(self.default_params, agent_id).n

    def action_space(self, params: EnvParams, agent_id: int):
        """Action space of the agent with id `agent_id`"""
        # We use the same action space for each agent
        # Both are Discrete, we set limit as the maximum of the two
        # And then enforce using action masking in agents and clipping in steps
        max_action = jnp.maximum(self.max_order_quantity, self.max_useful_life)
        return spaces.Discrete(max_action + 1)

    def observation_space(self, params: EnvParams, agent_id: int = 1) -> spaces.Box:
        """Observation space of the agent with id `agent_id`. For now, both the same"""
        # TODO: For not this is just the shape of the flat space, EnvObs.obs
        return spaces.Box(
            low=0,
            high=self.max_order_quantity,
            shape=(self.lead_time - 1 + self.max_useful_life),
            dtype=jnp.int32,
        )

    def state_space(self, params: EnvParams, agent_id: int) -> spaces.Dict:
        """State space of the environment."""
        # TODO: Infos is currently wrong
        return spaces.Dict(
            {
                "stock": spaces.Box(
                    low=0,
                    high=self.max_order_quantity,
                    shape=(self.max_useful_life),
                    dtype=jnp.int32,
                ),
                "in_transit": spaces.Box(
                    low=0,
                    high=self.max_order_quantity,
                    shape=(self.lead_time),
                    dtype=jnp.int32,
                ),
                "remaining_demand": spaces.Discrete(self.max_demand + 1),
                "agent_id": spaces.Discete(2),
                "cumulative_rewards": spaces.Box(
                    low=0, high=1e10, shape=(2,), dtype=jnp.int32
                ),
                "infos": spaces.Box(
                    low=0,
                    high=1e10,
                    shape=(len(self.agent_ids), len(self.info_idxs)),
                    dtype=jnp.int32,
                ),
                "time": spaces.Box(low=0, high=1e10, shape=(1,), dtype=jnp.int32),
            }
        )

    def _age_stock(self, state: EnvState, params: EnvParams) -> EnvState:
        """Ages stock by one time period and calculates expiry and holding cost."""
        stock, in_transit, infos, cumulative_rewards = (
            state.stock,
            state.in_transit,
            state.infos,
            state.cumulative_rewards,
        )
        # Age the stock by one day and calculate wastage cost
        expired = stock[self.max_useful_life - 1]
        infos = infos.replace(expiries=infos.expiries.at[:].add(expired))
        wastage_cost = expired * -params.wastage_cost
        cumulative_rewards = cumulative_rewards + wastage_cost

        # Age stock
        stock = jnp.roll(stock, shift=1)
        stock = stock.at[0].set(0)

        # Calculate holding cost
        holding = stock.sum()
        infos = infos.replace(holding=infos.holding.at[:].add(holding))
        holding_cost = holding * -params.holding_cost
        cumulative_rewards = cumulative_rewards + holding_cost

        # Receive the units ordered lead_time days ago
        stock = stock.at[0].set(in_transit[self.lead_time - 1])
        in_transit = jnp.roll(in_transit, shift=1)
        in_transit = in_transit.at[0].set(0)

        state = state.replace(
            stock=stock,
            in_transit=in_transit,
            infos=infos,
            cumulative_rewards=cumulative_rewards,
        )
        return state

    def _replenishment_step(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> EnvState:
        """Replenishment action step."""
        stock, in_transit, infos, cumulative_rewards = (
            state.stock,
            state.in_transit,
            state.infos,
            state.cumulative_rewards,
        )
        order = jnp.clip(action, a_min=0, a_max=self.max_order_quantity)
        infos = infos.replace(
            orders=infos.orders.at[:].add(order),
            order_placed=infos.orders.at[:].add(
                jax.lax.cond(order > 0, lambda: 1, lambda: 0)
            ),
        )

        # Place the order
        variable_order_cost = order * -params.variable_order_cost
        cumulative_rewards = cumulative_rewards + variable_order_cost
        in_transit = in_transit.at[0].set(order)

        # Sample demand from a truncated Gamma distribution
        demand_dist = distrax.Gamma(
            concentration=params.demand_gamma_alpha, rate=params.demand_gamma_beta
        )
        remaining_demand = (
            jnp.round(
                demand_dist.sample(seed=key)
            )  # Round because Gamma is continuous and demand discrete
            .clip(0, params.max_demand)  # Truncate at max demand
            .astype(jnp_int)  # Covert to integer
        )

        # Create updated state to return
        state = state.replace(
            stock=stock,
            in_transit=in_transit,
            infos=infos,
            cumulative_rewards=cumulative_rewards,
            remaining_demand=remaining_demand,
        )
        return state

    def _issuing_step(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> EnvState:
        stock, remaining_demand, infos, cumulative_rewards = (
            state.stock,
            state.remaining_demand,
            state.infos,
            state.cumulative_rewards,
        )

        # If action is above viable range, we set to 0 (nothing issued)
        action = jax.lax.select(action > self.max_useful_life, 0, action)

        # Meeting one unit of demand
        remaining_demand -= 1
        infos = infos.replace(demand=infos.demand.at[:].add(1))
        shortage = jnp.where(
            jax.lax.bitwise_or(stock[action - 1] == 0, action == 0), 1, 0
        )
        infos = infos.replace(shortages=infos.shortages.at[:].add(shortage))
        shortage_cost = shortage * -params.shortage_cost
        cumulative_rewards = cumulative_rewards + shortage_cost

        # Idx action - 1 because action is 1-indexed; we use 0 to indicate no unit issued
        # Issue one unit if there's no shortage (which includes our choice not to issue from stock)
        # TODO: i would rater use shortage in this cond, but getting an error
        stock = jax.lax.cond(
            shortage < 1,
            lambda stock, action: self._issue_one_unit(stock, action),
            lambda stock, action: stock,
            stock,
            action,
        )

        # Create updated state to return
        state = state.replace(
            stock=stock,
            infos=infos,
            cumulative_rewards=cumulative_rewards,
            remaining_demand=remaining_demand,
        )
        return state

    def _issue_one_unit(self, stock: jnp.array, action: int) -> chex.Array:
        """Issue one unit of stock."""
        # Clip to ensure we don't go below zero if there is a shortage
        stock = jax.lax.dynamic_update_index_in_dim(
            stock,
            jnp.clip(stock[action - 1] - 1, a_min=0, a_max=None),
            action - 1,
            axis=0,
        )
        return stock
