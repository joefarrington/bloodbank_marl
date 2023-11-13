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


class FlaxStochasticPolicy:
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
        self.env_params = default_env_params.replace(**env_params)
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)
        self.n_actions = int(env.num_actions(policy_id))
        self.model = model_class(n_actions=self.n_actions, **model_kwargs)

    def apply(self, policy_params, obs, rng):
        # Apply should get you an action
        pi, _ = self.model.apply(policy_params[self.policy_id], obs)
        tr_action = self._sample_action(pi, rng)
        return self.postprocess_action(obs, tr_action)

    def apply_for_training(self, policy_params, obs, rng):
        # Apply training should sample an action and return action, log_prob and value
        # In training, this should help us avoid issues around different action spaces
        # as long as both Discrete/both Box of same length etc.
        pi, value = self.model.apply(policy_params[self.policy_id], obs)
        tr_action = self._sample_action(pi, rng)
        log_prob = self._get_log_prob(pi, tr_action)
        action = self._postprocess_action(obs, tr_action)
        return action, tr_action, log_prob, value

    def apply_deterministic(self, policy_params, obs, rng):
        # Get the most likely action
        pi, _ = self.model.apply(policy_params[self.policy_id], obs)
        tr_action = self._get_mode_action(pi)
        return self._postprocess_action(obs, tr_action)

    def apply_for_loss_fn(self, policy_params, obs, tr_action):
        # Get the log prob of the action
        pi, value = self.model.apply(policy_params[self.policy_id], obs)
        log_prob = self._get_log_prob(pi, tr_action)
        entropy = pi.entropy().mean()
        return log_prob, value, entropy

    def get_initial_params(self, rng):
        return self.model.init(rng, self.obs)

    def _postprocess_action(self, obs, tr_action):
        return tr_action

    def _sample_action(self, pi, rng):
        return pi.sample(seed=rng)

    def _get_log_prob(self, pi, tr_action):
        return pi.log_prob(tr_action)

    def _get_mode_action(self, pi):
        return pi.mode()


class FlaxStochasticOrderUpToRepPolicy(FlaxStochasticPolicy):
    def _postprocess_action(self, obs, tr_action):
        S = tr_action
        action = jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        )
        return action


class FlaxMultiProductStochasticIssuePolicy(FlaxStochasticPolicy):
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


class FlaxMultiProductStochasticRepPolicy(FlaxStochasticPolicy):
    def __init__(
        self,
        model_class,
        model_kwargs,
        policy_id,
        env_name=None,
        env_kwargs={},
        env_params={},
        clip_min=-1,
        clip_max=1,
        min_order_quantity=0,
    ):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.replace(**env_params)
        self.model = model_class(n_actions=env.num_actions(policy_id), **model_kwargs)
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


class FlaxMultiProductStochasticOrderUpToRepPolicy(FlaxMultiProductStochasticRepPolicy):
    def _postprocess_action(self, obs, tr_action):
        S = super()._postprocess_action(obs, tr_action)
        action = jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        )
        return action
