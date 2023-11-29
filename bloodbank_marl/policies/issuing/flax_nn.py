from bloodbank_marl.scenarios.meneses_perishable.jax_env import (
    MenesesPerishableEnv,
    EnvObs,
)
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import numpy as np
import chex
from flax import struct
from bloodbank_marl.utils.gymnax_fitness import make
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import EnvObs
from bloodbank_marl.policies.common import FlaxStochasticMAPolicy


class FlaxStochasticMultiProductIssuePolicy(FlaxStochasticMAPolicy):
    def _sample_action(self, pi, rng):
        raw_action = pi.sample(seed=rng)
        tr_action = jnp.zeros(self.env_kwargs["n_products"])
        tr_action = jax.lax.select(
            raw_action == 0, tr_action, tr_action.at[raw_action - 1].add(1)
        )
        return tr_action

    def _get_log_prob(self, pi, tr_action):
        raw_action = jax.lax.select(tr_action.sum() == 0, 0, tr_action.argmax() + 1)
        return pi.log_prob(raw_action)

    def _get_mode_action(self, pi):
        raw_action = pi.mode()
        tr_action = jnp.zeros(self.env_kwargs["n_products"])
        tr_action = jax.lax.select(
            raw_action == 0, tr_action, tr_action.at[raw_action - 1].add(1)
        )
        return tr_action

    def _postprocess_action(self, obs, tr_action):
        return tr_action.astype(jnp.int32)
