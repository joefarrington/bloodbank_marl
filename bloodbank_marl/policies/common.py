import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import distrax
import numpy as np
import chex
from flax import struct
from bloodbank_marl.utils.gymnax_fitness import make


# TODO: Preprocess Obs/Postprocess action
# TODO: Should preprocess obs produce a flat representation?
# TODO: Order of arguments to init


class Policy:
    def __init__(
        self,
        env_name=None,
        env_kwargs={},
        env_params={},
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.create_env_params(**env_params)
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)

    def apply(self, policy_params, obs, rng):
        tr_action = self._apply(policy_params, obs, rng)
        action = self._postprocess_action(obs, tr_action)
        return action

    def _apply(self, policy_params, obs, rng):
        raise NotImplementedError

    def _postprocess_action(self, obs, tr_action):
        return tr_action

    #### Methods to make PPO training easier ####
    def apply_for_training(self, policy_params, obs, rng):
        tr_action = self._apply(policy_params, obs, rng)
        action = self._postprocess_action(obs, tr_action)
        # With a stochastic policy, second two terms are log_prob and value
        return action.astype(jnp.int32), tr_action.astype(jnp.float32), 0.0, 0.0

    def apply_deterministic(self, policy_params, obs, rng):
        # With a stochastic policy, this should return the most likely action
        return self.apply(policy_params, obs, rng)


class HeuristicPolicy(Policy):
    def __init__(
        self,
        env_name=None,
        env_kwargs={},
        env_params={},
        fixed_policy_params=None,  # Enables us to fix policy params for convenience
    ):
        super().__init__(
            env_name=env_name,
            env_kwargs=env_kwargs,
            env_params=env_params,
        )

        if fixed_policy_params is not None:
            self.fixed_policy_params = fixed_policy_params
            self.apply = self.apply_fixed
            self.apply_for_training = self.apply_for_training_fixed

    def apply_fixed(self, policy_params, obs, rng):
        """Accepts policy params as argument, but ignores them."""
        tr_action = self._apply(self.fixed_policy_params, obs, rng)
        action = self._postprocess_action(obs, tr_action)
        return action.astype(jnp.int32)

    def apply_for_training_fixed(self, policy_params, obs, rng):
        tr_action = self._apply(self.fixed_policy_params, obs, rng)
        action = self._postprocess_action(obs, tr_action)
        # With a stochastic policy, second two terms are log_prob and value
        return action.astype(jnp.int32), tr_action.astype(jnp.float32), 0.0, 0.0


class FlaxPolicy(Policy):
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
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)
        self.model = model_class(n_actions=env.num_actions, **model_kwargs)

    def _apply(self, policy_params, obs, rng):
        return self.model.apply(policy_params, obs, rng)

    def get_initial_params(self, rng):
        return self.model.init(rng, self.obs)


class FlaxMAPolicy(FlaxPolicy):
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
        self.model = model_class(
            n_actions=env.num_actions(policy_id),
            action_pad=env.action_padding(policy_id),
            **model_kwargs,
        )

    def _apply(self, policy_params, obs, rng):
        return self.model.apply(policy_params[self.policy_id], obs, rng)


class FlaxStochasticPolicy(FlaxPolicy):
    def _apply(self, policy_params, obs, rng):
        pi, _ = self.model.apply(policy_params, obs)
        tr_action = self._sample_action(pi, rng)
        return tr_action

    def apply_for_training(self, policy_params, obs, rng):
        # Apply training should sample an action and return action, log_prob and value
        # In training, this should help us avoid issues around different action spaces
        # as long as both Discrete/both Box of same length etc.
        pi, value = self.model.apply(policy_params, obs)
        tr_action = self._sample_action(pi, rng)
        log_prob = self._get_log_prob(pi, tr_action)
        action = self._postprocess_action(obs, tr_action)
        return action.astype(jnp.int32), tr_action.astype(jnp.float32), log_prob, value

    def apply_deterministic(self, policy_params, obs, rng):
        # Get the most likely action
        pi, _ = self.model.apply(policy_params, obs)
        tr_action = self._get_mode_action(pi)
        return self._postprocess_action(obs, tr_action)

    def apply_for_loss_fn(self, policy_params, obs, tr_action):
        # Get the log prob of the action
        pi, value = self.model.apply(policy_params, obs)
        log_prob = self._get_log_prob(pi, tr_action)
        entropy = pi.entropy().mean()
        return log_prob, value, entropy

    def _sample_action(self, pi, rng):
        return pi.sample(seed=rng)

    def _get_log_prob(self, pi, tr_action):
        return pi.log_prob(tr_action)

    def _get_mode_action(self, pi):
        return pi.mode()


class FlaxStochasticMAPolicy(FlaxStochasticPolicy):
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
            n_actions=self.n_actions, action_pad=self.action_pad, **model_kwargs
        )

    def _apply(self, policy_params, obs, rng):
        pi, _ = self.model.apply(policy_params[self.policy_id], obs)
        tr_action = self._sample_action(pi, rng)
        return tr_action

    def apply_for_training(self, policy_params, obs, rng):
        # Apply training should sample an action and return action, log_prob and value
        # In training, this should help us avoid issues around different action spaces
        # as long as both Discrete/both Box of same length etc.
        pi, value = self.model.apply(policy_params[self.policy_id], obs)
        tr_action = self._sample_action(pi, rng)
        log_prob = self._get_log_prob(pi, tr_action)
        action = self._postprocess_action(obs, tr_action)
        return action.astype(jnp.int32), tr_action.astype(jnp.float32), log_prob, value

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
