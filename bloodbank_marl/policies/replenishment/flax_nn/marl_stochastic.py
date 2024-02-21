import jax
import jax.numpy as jnp
from bloodbank_marl.policies.common import (
    FlaxStochasticPolicy,
    FlaxStochasticMAPolicy,
)
from bloodbank_marl.utils.gymnax_fitness import make


class FlaxStochasticOrderUpToRepPolicy(FlaxStochasticMAPolicy):
    def _postprocess_action(self, obs, tr_action):
        S = tr_action
        action = jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        )
        return action


class FlaxStochasticMultiProductRepPolicy(FlaxStochasticMAPolicy):
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
        self.env_params = default_env_params.create_env_params(**env_params)
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)
        self.n_actions = env.num_actions(policy_id)
        self.action_pad = env.action_padding(policy_id)
        self.model = model_class(
            n_actions=self.n_actions, action_pad=self.action_pad, **model_kwargs
        )
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
            )
            * obs.action_mask
        ).astype(jnp.int32)

        return action


class FlaxStochasticMultiProductOrderUpToRepPolicy(FlaxStochasticMultiProductRepPolicy):
    def _postprocess_action(self, obs, tr_action):
        S = super()._postprocess_action(obs, tr_action)
        action = jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        )
        return action
