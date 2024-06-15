# Adapted from  https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py

import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    # NOTE: This is customised for our setup, so could give it a more specific name

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, truncation, termination, _ = self._env.step(
            key, state.env_state, action, params
        )
        done = jax.lax.eq(env_state.live_agents.sum(axis=-1), 0)
        new_episode_return = state.episode_returns + (
            reward
            * (1 - env_state.agent_id)
            * (params.gamma ** jnp.clip(env_state.day - 1, 0, None))
        )  # So we only capture the reward of the first agent
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        # During training rollouts, we don't need to collect the info returned
        # by the environment, so we start from an empty dictionary here
        info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, truncation, termination, info


@struct.dataclass
class LogInfo:
    timestep: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_episode: bool
