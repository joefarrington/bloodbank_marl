import jax
import chex
from typing import Tuple, Union, Optional
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
class EnvState:
    stock: chex.Array
    in_transit: chex.Array
    remaining_demand: int
    agent_id: int  # Agent to act
    cumulative_rewards: chex.Array
    infos: chex.Array
    truncations: chex.Array
    terminations: chex.Array
    live_agents: chex.Array
    day: int  # Measures days
    step: int  # Measure total steps, rep and alloc combined


@struct.dataclass
class EnvObs:
    agent_id: int  # Following Tianhou;
    obs: chex.Array
    # mask: chex.Array # TODO: Action masking


class DeMoorPerishableMAJAX(object):
    """Jittable abstract base class for all gymnax-inspired Multi-Agent Environments."""

    def __init__(
        self, max_useful_life: int = 2, lead_time: int = 1, max_order_quantity: int = 10
    ):
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantity = max_order_quantity

        self.info_idxs = {
            "holding": 0,
            "wastage": 1,
            "shortage": 2,
            "demand": 3,
            "order": 4,
        }

        self.agent_ids = {"replenishment": 0, "issuing": 1}

        # self.possible_agents = ["replenishment", "issuing"]

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
        obs_st, state_st, reward, truncation, termination, info = self.step_env(
            key, state, action, params
        )
        # We're now only done if there are no live agents
        done = jax.lax.eq(state.live_agents.sum(), 0)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state, reward, truncation, termination, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[EnvParams] = None
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        obs, state = self.reset_env(key, params)
        return obs, state

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Environment-specific step transition."""

        # Get the current agent
        agent_id = state.agent_id

        # Zero the cumulative reward and reset the info for the agent that just stepped
        cumulative_rewards = jnp.where(
            jnp.arange(2) == state.agent_id, 0, state.cumulative_rewards
        )
        infos = state.infos.at[state.agent_id, :].set(0)
        state = EnvState(
            state.stock,
            state.in_transit,
            state.remaining_demand,
            state.agent_id,
            cumulative_rewards,
            infos,
            state.truncations,
            state.terminations,
            state.live_agents,
            state.day,
            state.step,
        )

        # TODO: Handle stepping an agent that is dead, as in PettingZoo
        return jax.lax.cond(
            jax.lax.bitwise_or(
                state.truncations[agent_id], state.terminations[agent_id]
            ),
            self.dead_step,
            self.live_step,
            key,
            state,
            action,
            params,
        )

    def dead_step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        live_agents = state.live_agents.at[state.agent_id].set(0)
        next_agent_id = jax.lax.cond(
            state.agent_id == 0, lambda: 1, lambda: 0
        )  # Switch to the other agent
        state = EnvState(
            state.stock,
            state.in_transit,
            state.remaining_demand,
            next_agent_id,
            state.cumulative_rewards,
            state.infos,
            state.truncations,
            state.terminations,
            live_agents,
            state.day,
            state.step + 1,
        )
        return (
            self.get_obs(state, next_agent_id),
            state,
            state.cumulative_rewards[next_agent_id],
            state.truncations[next_agent_id],
            state.terminations[next_agent_id],
            self.get_info(state, next_agent_id),
        )

    def live_step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        # Zero the cumulative reward and reset the info for the agent that just stepped
        cumulative_rewards = jnp.where(
            jnp.arange(2) == state.agent_id, 0, state.cumulative_rewards
        )
        infos = state.infos.at[state.agent_id, :].set(0)
        state = EnvState(
            state.stock,
            state.in_transit,
            state.remaining_demand,
            state.agent_id,
            cumulative_rewards,
            infos,
            state.truncations,
            state.terminations,
            state.live_agents,
            state.day,
            state.step,
        )

        # TODO: Difference between truncation and termination
        state, reward = jax.lax.switch(
            state.agent_id,
            [self._replenishment_step, self._issuing_step],
            key,
            state,
            action,
            params,
        )
        cumulative_rewards = cumulative_rewards + reward

        # Select the next agent
        # If no remainine demand, next agent is replenishment (agent_id 0)
        next_agent_id = jax.lax.cond(state.remaining_demand == 0, lambda: 0, lambda: 1)

        # If the next agent is replenishent, we need to age the stock, work out the holding cost, update the time
        state, reward = jax.lax.cond(
            next_agent_id == 0,
            self._age_stock,
            lambda state, params: (state, jnp.array([0.0, 0.0])),
            state,
            params,
        )
        cumulative_rewards = cumulative_rewards + reward
        day_increment = jax.lax.cond(next_agent_id == 0, lambda: 1, lambda: 0)

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
        state = EnvState(
            state.stock,
            state.in_transit,
            state.remaining_demand,
            next_agent_id,
            cumulative_rewards,
            state.infos,
            truncations,
            state.terminations,
            live_agents,
            state.day + day_increment,
            state.step + 1,
        )

        return (
            self.get_obs(state, next_agent_id),
            state,
            cumulative_rewards[next_agent_id],
            state.truncations[next_agent_id],
            state.terminations[next_agent_id],
            self.get_info(state, next_agent_id),
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Environment-specific reset."""
        state = EnvState(
            stock=jnp.zeros(self.max_useful_life, dtype=jnp.int32),
            in_transit=jnp.zeros(self.lead_time, dtype=jnp.int32),
            remaining_demand=0,
            agent_id=0,
            cumulative_rewards=jnp.array([0.0, 0.0]),
            infos=jnp.zeros(
                (len(self.agent_ids), len(self.info_idxs)), dtype=jnp.int32
            ),
            truncations=jnp.array([False] * len(self.agent_ids)),
            terminations=jnp.array([False] * len(self.agent_ids)),
            live_agents=jnp.array([1] * len(self.agent_ids)),
            day=0,
            step=0,
        )
        return self.get_obs(state, state.agent_id), state

    def get_obs(self, state: EnvState, agent_id: int) -> chex.Array:
        """Applies observation function to state, in PettinZoo AECEnv the equivalent is .observe()"""
        # TODO: For now, each agent gets the same observation and we'll deal with it at the agent level
        return EnvObs(
            state.agent_id,
            jnp.hstack([state.in_transit[1 : self.lead_time + 1], state.stock]),
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

    def discount(self, state: EnvState, params: EnvParams) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        return "DeMoorPerishable"

    @property
    def num_actions(self, agent_id: int) -> int:
        """Number of actions possible in environment for agent with id `agent_id`."""
        # TODO Add in number of actions
        raise NotImplementedError

    def action_space(self, params: EnvParams, agent_id: int):
        """Action space of the agent with id `agent_id`"""
        rep_space = spaces.Discrete(self.max_order_quantity + 1)
        issue_space = spaces.Discrete(self.max_useful_life + 1)
        return jax.lax.switch(agent_id, [lambda: rep_space, lambda: issue_space])

    def observation_space(self, params: EnvParams, agent_id: int = 1):
        """Observation space of the agent with id `agent_id`. For now, both the same"""
        return spaces.Box(
            low=0,
            high=self.max_order_quantity,
            shape=(self.lead_time - 1 + self.max_useful_life),
            dtype=jnp.int32,
        )

    def state_space(self, params: EnvParams, agent_id: int):
        """State space of the environment."""
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

    def _age_stock(
        self, state: EnvState, params: EnvParams
    ) -> Tuple[EnvState, chex.Array]:
        """Ages stock by one time period and calculates expiry and holding cost."""
        stock, in_transit, infos = state.stock, state.in_transit, state.infos
        # Age the stock by one day and calculate wastage cost
        expired = stock[self.max_useful_life - 1]
        infos = infos.at[:, self.info_idxs["wastage"]].add(expired)
        wastage_cost = expired * -params.wastage_cost

        # Age stock
        stock = jnp.roll(stock, shift=1)
        stock = stock.at[0].set(0)

        # Calculate holding cost
        holding = stock.sum()
        infos = infos.at[:, self.info_idxs["holding"]].add(holding)
        holding_cost = holding * -params.holding_cost

        # Receive the units ordered lead_time days ago
        stock = stock.at[0].set(in_transit[self.lead_time - 1])
        in_transit = jnp.roll(in_transit, shift=1)
        in_transit = in_transit.at[0].set(0)

        state = EnvState(
            stock,
            in_transit,
            state.remaining_demand,
            state.agent_id,
            state.cumulative_rewards,
            infos,
            state.truncations,
            state.terminations,
            state.live_agents,
            state.day,
            state.step,
        )
        total_cost = wastage_cost + holding_cost
        reward = jnp.array([total_cost, total_cost])
        return state, reward

    def _replenishment_step(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[EnvState, chex.Array]:
        """Replenishment action step."""
        in_transit, infos = state.in_transit, state.infos
        order = jnp.clip(action, a_min=0, a_max=self.max_order_quantity)
        infos = infos.at[self.agent_ids["replenishment"], self.info_idxs["order"]].add(
            order
        )

        # Place the order
        variable_order_cost = order * -params.variable_order_cost
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
        infos = infos.at[self.agent_ids["replenishment"], self.info_idxs["demand"]].add(
            remaining_demand
        )

        # Create updated state to return
        state = EnvState(
            state.stock,
            in_transit,
            remaining_demand,
            state.agent_id,
            state.cumulative_rewards,
            infos,
            state.truncations,
            state.terminations,
            state.live_agents,
            state.day,
            state.step,
        )
        reward = jnp.array([variable_order_cost, variable_order_cost])
        return state, reward

    def _issuing_step(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[EnvState, chex.Array]:
        stock, remaining_demand, infos = (
            state.stock,
            state.remaining_demand,
            state.infos,
        )
        # Meeting one unit of demand
        remaining_demand -= 1
        infos = infos.at[self.agent_ids["issuing"], self.info_idxs["demand"]].add(1)
        shortage = jnp.where(
            jax.lax.bitwise_or(stock[action - 1] == 0, action == 0), 1, 0
        )
        infos = infos.at[:, self.info_idxs["shortage"]].add(shortage)
        shortage_cost = shortage * -params.shortage_cost

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
        state = EnvState(
            stock,
            state.in_transit,
            remaining_demand,
            state.agent_id,
            state.cumulative_rewards,
            infos,
            state.truncations,
            state.terminations,
            state.live_agents,
            state.day,
            state.step,
        )
        reward = jnp.array([shortage_cost, shortage_cost])
        return state, reward

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

    def get_info(self, state: EnvState, agent_id: int) -> chex.Array:
        """Returns info for the current agent"""
        return {k: state.infos[agent_id][v] for k, v in self.info_idxs.items()}

    ## These are methods from PettingZoo that we're not currently using
    ## Useful to think about if doing some sort of PettingZoo wrapper
    def last(self, state: EnvState, observe: bool = True) -> chex.Array:
        """Returns observation, cumulative reward, terminated and info for the current agent"""
        raise NotImplementedError  # Used in PettingZooEnv

    def _clear_reward(self, state) -> EnvState:
        """Clears all rewards from state."""
        raise NotImplementedError

    def _accumulate_rewards(self, state) -> EnvState:
        """Adds rewards to cumulvative rewards, typically called near the end of .step()"""
        raise NotImplementedError
