from bloodbank_marl.policies.common import HeuristicPolicy
import jax
import jax.numpy as jnp
import numpy as np
from typing import List
from bloodbank_marl.utils.gymnax_fitness import make


class VIPolicy(HeuristicPolicy):
    def _apply(self, policy_params, obs, rng):
        return policy_params[tuple(obs.obs)]
