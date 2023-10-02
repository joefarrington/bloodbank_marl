import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Dict, Optional, Any
from bloodbank_marl.utils.gymnax_fitness import make
import pandas as pd


class FlaxRepPolicy:
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
            n_actions=env.max_order_quantity + 1, **policy_kwargs
        )

    def get_params(self, rng):
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        obs = jnp.zeros(env.observation_space(self.env_params, 0).shape)
        return self.policy_net.init(rng, obs)

    def apply(self, policy_params, obs, rng):
        return self.policy_net.apply(policy_params[self.policy_id], obs, rng)


class RepMLP(nn.Module):
    n_hidden: int
    n_actions: int  # For now, number of actions is the max_order_quantity + 1

    @nn.compact
    def __call__(self, x, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        s = jnp.argmax(x, axis=-1)
        return s


# Order up to policy function for use with FixedPolicy
def order_up_to(policy_params, obs, rng, env_kwargs):
    return jnp.clip(
        policy_params - jnp.sum(obs), a_min=0, a_max=env_kwargs["max_order_quantity"]
    )


# Policy to apply actions already learned from value iteration
# Modified from viso_jax//policies/value_iteration_policy.py
# TODO: We need to think about how best to align this and the order_up_to/FIFO/LIFO for use
# with FIxedPolicy so that they can all easily be used with a config file.
class VIPolicy(object):
    def __init__(
        self,
        env_name: str,
        env_kwargs: Optional[Dict[str, Any]] = {},
        env_params: Optional[Dict[str, Any]] = {},
        agent_id: Optional[
            int
        ] = 0,  # The observation space takes the agent_id as an argument so we need to specify which agent we're setting up
        policy_params_filepath: Optional[str] = None,
        policy_params_df: Optional[pd.DataFrame] = None,
    ):
        # TODO: Create new make function and register envs
        # if env_id not in registered_envs:
        #    raise ValueError("Environment ID is not registered.")
        if (policy_params_filepath is not None) and (policy_params_df is not None):
            raise ValueError(
                "Supply policy parameters using only one of policy_params_filepath or policy_params_df"
            )
        elif (policy_params_filepath is not None) or (policy_params_df is not None):
            env, default_env_params = make(env_name, **env_kwargs)
            env_params = default_env_params.create_env_params(**env_params)
            self.obs_space_shape = jnp.hstack(
                [
                    env.observation_space(env_params, agent_id).high
                    - env.observation_space(env_params, agent_id).low
                    + 1,
                    -1,  # Shape of action, will be squeezed out if 1
                ]
            )
            if policy_params_filepath:
                policy_params_df = self._load_policy_params_df(policy_params_filepath)
            self.policy_params = self._policy_params_df_to_array(policy_params_df)

    def apply(self, policy_params, obs, rng, env_kwargs):
        order = policy_params[tuple(obs)]
        return order

    def get_params(self, rng=None):
        return self.policy_params

    def _load_policy_params_df(self, filepath):
        policy_params_df = pd.read_csv(filepath, index_col=0)
        return policy_params_df

    def _policy_params_df_to_array(self, policy_params_df):
        policy_params = jnp.array(
            policy_params_df.values.reshape(self.obs_space_shape)
        ).squeeze()
        return policy_params
