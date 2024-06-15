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
from bloodbank_marl.utils.make_env import make
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import EnvObs
from bloodbank_marl.policies.common import FlaxStochasticMAPolicy, FlaxPolicy


class FlaxMultiProductIssuePolicySingleAgentEnv(FlaxPolicy):
    # For use with the new environment - needs to get an IssueObs to instantiate NN params
    def __init__(
        self,
        model_class,
        model_kwargs,
        env_name=None,
        env_kwargs={},
        env_params={},
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.create_env_params(**env_params)
        self.obs = env.default_issue_obs(self.env_params)
        self.model = model_class(
            n_actions=env.num_issue_actions + 1, **model_kwargs
        )  # Adding one for the no-issue action


class FlaxStochasticMultiProductIssuePolicy(FlaxStochasticMAPolicy):
    def __init__(
        self,
        model_class,
        model_kwargs,
        policy_id,
        env_name=None,
        env_kwargs={},
        env_params={},
    ):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.create_env_params(**env_params)
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)
        self.n_actions = int(env.num_actions(policy_id))
        self.action_pad = int(env.action_padding(policy_id))
        self.model = model_class(
            n_actions=self.n_actions + 1, action_pad=self.action_pad, **model_kwargs
        )  # Add one to reflect issuing nothing; gets resshaped to nb_products in methods that get the action

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
