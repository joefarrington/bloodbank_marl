import jax
import jax.numpy as jnp
import flax.linen as nn
import chex
from typing import Dict, Optional, Any, List
from bloodbank_marl.utils.gymnax_fitness import make
from bloodbank_marl.utils.yaml import from_yaml, to_yaml
from bloodbank_marl.environments.environment import MarlEnvironment
from bloodbank_marl.scenarios.meneses_perishable.jax_env import (
    MenesesPerishableEnv,
    EnvObs,
    EnvParams,
    EnvInfo,
    EnvState,
)
import numpy as np
import pandas as pd


class FlaxRepPolicy:
    def __init__(
        self,
        model_class,
        model_kwargs,
        policy_id,
        env_name,
        env_kwargs={},
        env_params={},
    ):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.replace(**env_params)
        self.model = model_class(
            n_actions=env.action_space(self.env_params, policy_id).n, **model_kwargs
        )

    def get_params(self, rng):
        env, _ = make(self.env_name, **self.env_kwargs)
        rng, _rng = jax.random.split(rng)
        obs, _ = env.reset(_rng, self.env_params)
        return self.model.init(rng, obs)

    def apply(self, policy_params, obs, rng):
        return self.model.apply(policy_params[self.policy_id], obs, rng)


class RepDiscreteMLP(nn.Module):
    n_hidden: int
    n_actions: int

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = obs.obs
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = x + jnp.where(obs.action_mask == 1, 0, -1e9)
        x = jnp.argmax(x, axis=-1)
        return x


class RepDiscreteOrderUpToMLP(nn.Module):
    # NOTE: This is fine for order-up-to for NeuroEvo because
    # we don't have to calculate the log_prob of the action
    # but it won't work for PPO because we need to calculate
    # the log_prob of the action. For that, we'd need some post-processing
    # of the action so that the S level can be stored in the transition but the
    # order quantity is given to the environment.

    n_hidden: int
    n_actions: int

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = obs.obs
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = x + jnp.where(obs.action_mask == 1, 0, -1e9)
        S = jnp.argmax(x, axis=-1)
        return jnp.clip(S - obs.obs.sum(axis=-1), a_min=0).astype(jnp.int32)


class RepMultiProductMLP(nn.Module):
    # NOTE: Don't need a distribution for NeuroEvo, so instead of sampling from Gaussian we're applying tanh
    # therefore clip_min is -1 and clip_max is +1 (and there won't be any actual clipping)
    n_hidden: int
    n_actions: int
    max_order_quantity: int
    min_order_quantity: int = 0
    clip_min = -1
    clip_max = 1

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = obs.obs
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = nn.tanh(x)
        x = self.map_to_discrete(x)
        x = (
            x * obs.action_mask
        )  # We assume that action mask just says whether we can order this product at all or no
        return x

    def map_to_discrete(self, x):
        clipped_outputs = jnp.clip(
            jnp.round(x), a_min=self.clip_min, a_max=self.clip_max
        )
        return jnp.ceil(
            (
                ((clipped_outputs - self.clip_min) / (self.clip_max - self.clip_min))
                * (self.max_order_quantity - self.min_order_quantity)
            )
            + self.min_order_quantity
        ).astype(jnp.int32)

class RepMultiProductOrderUpToMLP(RepMultiProductMLP):
    # NOTE: As with the discrete one, this is fine for order-up-to for NeuroEvo because
    # we don't have to calculate the log_prob of the action
    # but it won't work for PPO because we need to calculate
    # the log_prob of the action. For that, we'd need some post-processing
    # of the action so that the S level can be stored in the transition but the
    # order quantity is given to the environment.


    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = obs.obs
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = nn.tanh(x)
        x = self.map_to_discrete(x)
        S = (
            x * obs.action_mask
        )  # We assume that action mask just says whether we can order this product at all or no
        return return jnp.clip(S - obs.obs.sum(axis=-1), a_min=0).astype(jnp.int32)

# Order up to policy function for use with FixedPolicy
def order_up_to(policy_params, obs, rng, env_kwargs):
    obs = obs.obs
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
        order = policy_params[tuple(obs.obs)]
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


class HeuristicPolicy:
    def __init__(
        self,
        env_name: str,
        env_kwargs: Optional[Dict[str, Any]] = {},
        env_params: Optional[Dict[str, Any]] = {},
        policy_params_filepath: Optional[str] = None,
    ):
        # As in utils/rollout.py env_kwargs and env_params arguments are dicts to
        # override the defaults for an environment.

        # Instantiate an internal envinronment we'll use to access env kwargs/params
        # These are not stored, just used to set up param_col_names, param_row_names and forward
        # TODO: Update make statement
        self.env_name = env_name
        env, default_env_params = make(self.env_name, **env_kwargs)
        all_env_params = default_env_params.replace(**env_params)

        self.param_col_names = self._get_param_col_names(env_name, env, all_env_params)
        self.param_row_names = self._get_param_row_names(env_name, env, all_env_params)
        self.apply = self._get_apply_method(env_name, env, all_env_params)

        if self.param_row_names != []:
            self.param_names = np.array(
                [
                    [f"{p}_{r}" for p in self.param_col_names]
                    for r in self.param_row_names
                ]
            )
        else:
            self.param_names = np.array([self.param_col_names])

        self.params_shape = self.param_names.shape

        if policy_params_filepath:
            self.policy_params = self.load_policy_params(policy_params_filepath)

    def _get_param_col_names(
        self, env_name: str, env: MarlEnvironment, env_params: Dict[str, Any]
    ) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        raise NotImplementedError

    def _get_param_row_names(
        self, env_name: str, env: MarlEnvironment, env_params: Dict[str, Any]
    ) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        raise NotImplementedError

    def _get_apply_method(
        self, env_name: str, env: MarlEnvironment, env_params: Dict[str, Any]
    ) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        raise NotImplementedError

    def load_policy_params(self, filepath: str) -> chex.Array:
        """Load policy parameters from a yaml file"""
        params_dict = from_yaml(filepath)["policy_params"]
        if self.param_row_names is None:
            params_df = pd.DataFrame(params_dict, index=[0])
        else:
            params_df = pd.DataFrame(params_dict)
        policy_params = jnp.array(params_df.values)
        assert (
            policy_params.shape == self.params_shape
        ), f"Parameters in file do not match expected shape: found {policy_params.shape} and expected {self.params_shape}"
        return policy_params


class SPolicy(HeuristicPolicy):
    def _get_param_col_names(
        self, env_name: str, env: MarlEnvironment, env_params: EnvParams
    ) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["S"]

    def _get_param_row_names(
        self, env_name: str, env: MarlEnvironment, env_params: EnvParams
    ) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if env_name == "MenesesPerishable" or "MenesesPerishableGymnax":
            return ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
        else:
            return []

    def _get_apply_method(
        self, env_name: str, env: MarlEnvironment, env_params: EnvParams
    ) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        if env_name == "DeMoorPerishable":
            return de_moor_perishable_S_policy
        elif env_name == "MenesesPerishable" or "MenesesPerishableGymnax":
            return meneses_perishable_S_policy
        else:
            raise NotImplementedError(
                f"No (S) policy defined for Environment ID {env_name}"
            )


def meneses_perishable_S_policy(policy_params, obs, rng):
    """S policy for scenario based on Meneses et al 2021"""
    stock_on_hand_and_in_transit = obs.stock.sum(
        axis=-1, keepdims=True
    ) + obs.in_transit.sum(axis=-1, keepdims=True)
    # Clip because we can#t have a negative order
    # Squeeze because action should be a 1D array
    return jnp.clip((policy_params - stock_on_hand_and_in_transit), a_min=0).squeeze()


def de_moor_perishable_S_policy(policy_params, obs, rng):
    """(S) policy for DeMoorPerishable environment"""
    # policy_params = [[S]]
    stock_on_hand_and_in_transit = obs.stock.sum() + obs.in_transit.sum()
    return jnp.clip((policy_params[0, 0] - stock_on_hand_and_in_transit), a_min=0)
