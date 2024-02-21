import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Dict, Optional, Any, List
from bloodbank_marl.utils.gymnax_fitness import make
from bloodbank_marl.utils.yaml import from_yaml, to_yaml
from bloodbank_marl.environments.environment import MarlEnvironment
from bloodbank_marl.scenarios.meneses_perishable.jax_env import (
    MenesesPerishableEnv,
    EnvObs,
    EnvParams,
    EnvInfo,
    EnvState,
)
import numpy as np
import pandas as pd


# We need to define policies slightly differently here because
# in gymnax env there is no policy id and no need for action padding
# when only a single agent
class FlaxRepPolicy:
    def __init__(
        self,
        model_class,
        model_kwargs,
        env_name,
        env_kwargs={},
        env_params={},
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.create_env_params(**env_params)
        self.model = model_class(
            n_actions=env.num_actions,
            **model_kwargs,
        )

    def get_params(self, rng):
        env, _ = make(self.env_name, **self.env_kwargs)
        rng, _rng = jax.random.split(rng)
        obs, _ = env.reset(_rng, self.env_params)
        return self.model.init(rng, obs)

    def apply(self, policy_params, obs, rng):
        raw_action = self.model.apply(policy_params, obs, rng)
        return self.postprocess_action(obs, raw_action)

    def postprocess_action(self, obs, raw_action):
        return raw_action


class FlaxMultiProductRepPolicy(FlaxRepPolicy):
    def __init__(
        self,
        model_class,
        model_kwargs,
        env_name,
        env_kwargs={},
        env_params={},
        clip_min=-1,
        clip_max=1,
        min_order_quantity=0,
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.create_env_params(**env_params)
        self.model = model_class(
            n_actions=env.num_actions,
            **model_kwargs,
        )
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.max_order_quantities = env.max_order_quantities
        self.min_order_quantities = (
            jnp.ones_like(self.max_order_quantities) * min_order_quantity
        )

    def postprocess_action(self, obs, raw_action):
        clipped_outputs = jnp.clip(raw_action, a_min=self.clip_min, a_max=self.clip_max)
        action = (
            jnp.ceil(
                (
                    (
                        (clipped_outputs - self.clip_min)
                        / (self.clip_max - self.clip_min)
                    )
                    * (self.max_order_quantities - self.min_order_quantities)
                )
                + self.min_order_quantities
            ).astype(jnp.int32)
            * obs.action_mask
        )
        return action


class FlaxMultiProductOrderUpToRepPolicy(FlaxMultiProductRepPolicy):
    def postprocess_action(self, obs, raw_action):
        S = super().postprocess_action(obs, raw_action)
        return jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        )


# This should be an easier case, where just have the total stock per product
# Might be nice to have a function in policy for obs preprocessing
class RepMultiProductTotalHoldingMLP(nn.Module):
    n_hidden: int
    n_actions: int
    action_pad: int = 0

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = obs.stock.sum(axis=-1) + obs.in_transit.sum(axis=-1)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.n_actions)(x)
        x = nn.tanh(x)
        return x
