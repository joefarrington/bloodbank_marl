import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Dict, Optional, Any, List
from bloodbank_marl.utils.make_env import make
from bloodbank_marl.utils.yaml import from_yaml, to_yaml
from bloodbank_marl.environments.marl_environment import MarlEnvironment
from bloodbank_marl.scenarios.meneses_perishable.jax_env import (
    MenesesPerishableEnv,
    EnvObs,
    EnvParams,
    EnvInfo,
    EnvState,
)
import numpy as np
import pandas as pd
from bloodbank_marl.policies.common import FlaxPolicy


class FlaxOrderUpToRepPolicy(FlaxPolicy):
    def _postprocess_action(self, obs, tr_action):
        S = tr_action
        return jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        ).astype(jnp.int32)


class FlaxMultiProductRepPolicy(FlaxPolicy):
    def __init__(
        self,
        model_class,
        model_kwargs,
        env_name,
        env_kwargs={},
        env_params={},
        clip_min=-2,
        clip_max=2,
        min_order_quantity=0,
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.create_env_params(**env_params)
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)
        self.model = model_class(n_actions=env.num_actions, **model_kwargs)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.max_order_quantities = env.max_order_quantities
        self.min_order_quantities = (
            jnp.ones_like(self.max_order_quantities) * min_order_quantity
        )

    def _postprocess_action(self, obs, tr_action):
        clipped_outputs = jnp.clip(tr_action, a_min=self.clip_min, a_max=self.clip_max)
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
    def _postprocess_action(self, obs, tr_action):
        S = super()._postprocess_action(obs, tr_action)
        return jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        )


class FlaxMultiProductCategoricalRepPolicy(FlaxPolicy):
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
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)
        # TODO: Note that this always takes the highest max order quantity
        # for now - could modify with some clipping etc if needed
        self.model = model_class(n_actions=env.num_actions, **model_kwargs)


class FlaxMultiProductOrderUpToCategoricalRepPolicy(
    FlaxMultiProductCategoricalRepPolicy
):
    def _postprocess_action(self, obs, tr_action):
        S = super()._postprocess_action(obs, tr_action)
        return jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        )
