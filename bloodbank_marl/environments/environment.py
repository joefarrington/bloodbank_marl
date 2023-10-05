import jax
import chex
from typing import Tuple, Union, Optional, Dict
from functools import partial
from flax import struct
import jax.numpy as jnp
from gymnax.environments import spaces
import numpy as np
import gymnax

jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


@struct.dataclass
class EnvInfo:
    @classmethod
    def create_empty_infos(cls):
        """Create an empty info object"""
        return NotImplemented

    def reset_infos_one_agent(self, agent_id: int):
        """Reset the infos for a single agent"""
        raise NotImplementedError

    def accumulate_infos_one_agent(self, agent_id: int, step_info):
        """Add step info to cumulative info for a single agent"""
        raise NotImplementedError


# We're using state to track both the state of the environment in MDP terms and also
# additional information that in, say, PettingZoo would be stored as attributes of the
# Env class like the next agent, cumulative rewards, etc.
# TODO: We COULD split this up a bit more - e.g. have a sub-dataclass just for the MDP state as
# we currently do for info. But, I think given use of EnvObs, this is fine.
@struct.dataclass
class EnvState:
    agent_id: int
    cumulative_rewards: chex.Array
    infos: EnvInfo
    truncations: chex.Array
    terminations: chex.Array
    live_agents: chex.Array
    step: int  # Gymnax uses time for step, we reserve time because in some envs we want a concept of time separate from step


@struct.dataclass
class EnvParams:
    max_steps_in_episode: int


# If it's helpful for a policy to have access to a more strcuture observation
# We can add extra fields, but always report at least a flat observation.
@struct.dataclass
class EnvObs:
    agent_id: int

    @property
    def obs(self):
        """Return the observation as a flat array."""
        raise NotImplementedError


class MarlEnvironment(object):
    """Jittable abstract base class for all gymnax-inspired Multi-Agent Environments."""

    def __init__(
        self,
        agent_names: Tuple[str, ...],
    ):
        """Initializes the environment."""
        self.possible_agents = agent_names
        self.agent_ids = {agent_name: i for i, agent_name in enumerate(agent_names)}
        self.n_agents = len(agent_names)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[EnvObs, EnvState, float, bool, bool, EnvInfo]:
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
    ) -> Tuple[EnvObs, EnvState]:
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
    ) -> Tuple[EnvObs, EnvState, float, bool, bool, EnvInfo]:
        """Environment-specific step transition."""

        # Get the current agent
        agent_id = state.agent_id

        cumulative_rewards, infos = state.cumulative_rewards, state.infos

        # Zero the cumulative reward and reset the info for the agent that just stepped
        cumulative_rewards = jnp.where(
            jnp.arange(len(self.agent_ids)) == state.agent_id,
            0,
            state.cumulative_rewards,
        )
        infos = infos.reset_infos_one_agent(agent_id)
        state = state.replace(cumulative_rewards=cumulative_rewards, infos=infos)

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
    ) -> Tuple[EnvObs, EnvState, float, bool, bool, EnvInfo]:
        live_agents = state.live_agents.at[state.agent_id].set(0)
        next_agent_id = jax.lax.cond(
            state.agent_id == 0, lambda: 1, lambda: 0
        )  # Switch to the other agent
        state = state.replace(
            agent_id=next_agent_id, live_agents=live_agents, step=state.step + 1
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
    ) -> Tuple[EnvObs, EnvState, float, bool, bool, EnvInfo]:
        raise NotImplementedError

        return (
            self.get_obs(state, next_agent_id),
            state,
            state.cumulative_rewards[next_agent_id],
            state.truncations[next_agent_id],
            state.terminations[next_agent_id],
            self.get_info(state, next_agent_id),
        )

    def get_info(self, state: EnvState, agent_id: int) -> EnvInfo:
        """Returns info for the current agent"""
        return jax.tree_map(lambda x: x[agent_id], state.infos)

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[EnvObs, EnvState]:
        """Environment-specific reset."""

        raise NotImplementedError

    def get_obs(self, state: EnvState, agent_id: int) -> EnvObs:
        """Applies observation function to state, in PettinZoo AECEnv the equivalent is .observe()"""

        raise NotImplementedError

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state transition is terminal."""

        raise NotImplementedError

    def discount(self, state: EnvState, params: EnvParams) -> float:
        """Return a discount of zero if the episode has terminated."""
        return jax.lax.select(self.is_terminal(state, params), 0.0, 1.0)

    @property
    def name(self) -> str:
        """Environment name."""
        raise NotImplementedError

    @property
    def num_actions(self, agent_id: int) -> int:
        """Number of actions possible in environment for agent with id `agent_id`."""
        # TODO Add in number of actions
        raise NotImplementedError

    def action_space(self, params: EnvParams, agent_id: int):
        """Action space of the agent with id `agent_id`"""
        raise NotImplementedError

    def observation_space(self, params: EnvParams, agent_id: int = 1):
        """Observation space of the agent with id `agent_id`. For now, both the same"""
        raise NotImplementedError

    def state_space(self, params: EnvParams, agent_id: int):
        """State space of the environment."""
        raise NotImplementedError

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
