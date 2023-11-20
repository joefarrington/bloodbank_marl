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
            n_actions=env.num_actions(policy_id),
            action_pad=env.action_padding(policy_id),
            **model_kwargs,
        )

    def get_params(self, rng):
        env, _ = make(self.env_name, **self.env_kwargs)
        rng, _rng = jax.random.split(rng)
        obs, _ = env.reset(_rng, self.env_params)
        return self.model.init(rng, obs)

    def apply(self, policy_params, obs, rng):
        raw_action = self.model.apply(policy_params[self.policy_id], obs, rng)
        return self.postprocess_action(obs, raw_action)

    def postprocess_action(self, obs, raw_action):
        return raw_action


class FlaxOrderUpToRepPolicy(FlaxRepPolicy):
    def postprocess_action(self, obs, raw_action):
        return jnp.clip(
            raw_action - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        ).astype(jnp.int32)


class FlaxMultiProductRepPolicy(FlaxRepPolicy):
    def __init__(
        self,
        model_class,
        model_kwargs,
        policy_id,
        env_name,
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
        self.model = model_class(
            n_actions=env.num_actions(policy_id),
            action_pad=env.action_padding(policy_id),
            **model_kwargs,
        )
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.max_order_quantities = env.max_order_quantities
        self.min_order_quantities = (
            jnp.ones_like(self.max_order_quantities) * min_order_quantity
        )

    def postprocess_action(self, obs, raw_action):
        clipped_outputs = jnp.clip(raw_action, a_min=self.clip_min, a_max=self.clip_max)
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


class FlaxMultiProductOrderUpToRepPolicy(FlaxMultiProductRepPolicy):
    def postprocess_action(self, obs, raw_action):
        S = super().postprocess_action(obs, raw_action)
        return jnp.clip(
            S - obs.stock.sum(axis=-1) - obs.in_transit.sum(axis=-1),
            a_min=0,
            a_max=None,
        )


class RepDiscreteMLP(nn.Module):
    n_hidden: int
    n_actions: int
    action_pad: int = 0

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = obs.obs
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = jnp.hstack([x, jnp.zeros((x.shape[:-1] + (self.action_pad,)))])
        x = x + jnp.where(obs.action_mask == 1, 0, -1e9)
        x = jnp.argmax(x, axis=-1)
        return x


class RepMultiProductMLP(nn.Module):
    n_hidden: int
    n_actions: int
    action_pad: int = 0

    @nn.compact
    def __call__(self, obs, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        # x = obs.stock.sum(axis=-1) # Alternative is just have total number of stock per product
        x = obs.obs
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_actions)(x)
        x = nn.tanh(x)
        return x


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
