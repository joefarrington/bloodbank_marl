import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Optional
from bloodbank_marl.utils.gymnax_fitness import make


class FlaxRepPolicy:
    def __init__(self, policy_class, policy_id, env_name, env_kwargs={}, env_params={}):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.env_params = env_params
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.policy_net = policy_class(env.max_order_quantity + 1)

    def get_params(self, rng):
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        obs = jnp.zeros(env.observation_space(self.env_params, 0).shape)
        return self.policy_net.init(rng, obs)

    def apply(self, policy_params, obs, rng):
        return self.policy_net.apply(policy_params[self.policy_id], obs, rng)


class RepMLP(nn.Module):
    n_actions: int  # For now, number of actions is the max_order_quantity + 1

    @nn.compact
    def __call__(self, x, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        s = jnp.argmax(x, axis=-1)
        return s


def order_up_to(policy_params, obs, rng, env_kwargs):
    return jnp.clip(
        policy_params - jnp.sum(obs), a_min=0, a_max=env_kwargs["max_order_quantity"]
    )
