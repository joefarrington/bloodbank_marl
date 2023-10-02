import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Optional
from bloodbank_marl.utils.gymnax_fitness import make


class FlaxIssuePolicy:
    def __init__(
        self,
        policy_class,
        policy_kwargs,
        policy_id,
        env_name,
        env_kwargs={},
        env_params={},
    ):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.env_params = env_params
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.policy_net = policy_class(
            n_actions=env.max_useful_life + 1, **policy_kwargs
        )

    def get_params(self, rng):
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        obs = jnp.zeros(env.observation_space(self.env_params, 0).shape)
        return self.policy_net.init(rng, obs)

    def apply(self, policy_params, obs, rng):
        return self.policy_net.apply(policy_params[self.policy_id], obs, rng)


class IssueMLP(nn.Module):
    n_hidden: int
    n_actions: int

    @nn.compact
    def __call__(self, x, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        s = jnp.argmax(x, axis=-1)
        return s


def issue_fifo(policy_params, obs, rng, env_kwargs):
    """Action is idx of oldest unit, plus one.
    Zero if no units are present"""
    stock = obs[
        env_kwargs["lead_time"]
        - 1 : env_kwargs["max_useful_life"]
        + env_kwargs["lead_time"]
        - 1
    ]
    return jax.lax.cond(
        jnp.sum(stock) == 0,
        lambda _: jnp.array(0),
        lambda _: env_kwargs["max_useful_life"]
        - jnp.flip(jnp.where(stock > 0, 1, 0)).argmax(),
        None,
    )


def issue_lifo(policy_params, obs, rng, env_kwargs):
    """Action is idx of youngest unit, plus one.
    Zero if no units are present"""
    stock = obs[
        env_kwargs["lead_time"]
        - 1 : env_kwargs["max_useful_life"]
        + env_kwargs["lead_time"]
        - 1
    ]
    return jax.lax.cond(
        jnp.sum(stock) == 0,
        lambda _: jnp.array(0),
        lambda _: 1 + (jnp.where(stock > 0, 1, 0)).argmax(),
        None,
    )
