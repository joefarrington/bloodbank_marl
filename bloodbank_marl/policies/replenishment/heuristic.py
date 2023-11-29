from bloodbank_marl.policies.common import HeuristicPolicy
import jax
import jax.numpy as jnp
import numpy as np
from typing import List
from bloodbank_marl.utils.gymnax_fitness import make


class SRepPolicy(HeuristicPolicy):
    def __init__(
        self,
        env_name=None,
        env_kwargs={},
        env_params={},
        fixed_policy_params=None,  # Enables us to fix policy params for convenience
    ):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.replace(**env_params)
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)

        # Set up parameters
        self._setup_param_properties()

        # Set up apply method
        self._apply = self._get_apply_method()

        if fixed_policy_params is not None:
            self.fixed_policy_params = fixed_policy_params
            self.apply = self.apply_fixed
            self.apply_for_training = self.apply_for_training_fixed

    def _setup_param_properties(self):
        self.param_col_names = self._get_param_col_names()
        self.param_row_names = self._get_param_row_names()
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

    def _get_param_col_names(self) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["S"]

    def _get_param_row_names(self) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if self.env_name in ["MenesesPerishable", "MenesesPerishableGymnax"]:
            return ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
        else:
            return []

    def _get_apply_method(self) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        if self.env_name == "DeMoorPerishable":
            return de_moor_perishable_S_policy
        elif self.env_name == "MenesesPerishable" or "MenesesPerishableGymnax":
            return meneses_perishable_S_policy
        else:
            raise NotImplementedError(
                f"No (S) policy defined for Environment ID {self.env_name}"
            )


def meneses_perishable_S_policy(policy_params, obs, rng):
    """S policy for scenario based on Meneses et al 2021"""
    stock_on_hand_and_in_transit = obs.stock.sum(
        axis=-1, keepdims=True
    ) + obs.in_transit.sum(axis=-1, keepdims=True)
    # Clip because we can't have a negative order
    # Squeeze because action should be a 1D array
    return jnp.clip((policy_params - stock_on_hand_and_in_transit), a_min=0).squeeze()


def de_moor_perishable_S_policy(policy_params, obs, rng):
    """(S) policy for DeMoorPerishable environment"""
    # policy_params = [[S]]
    stock_on_hand_and_in_transit = obs.stock.sum() + obs.in_transit.sum()
    return jnp.clip((policy_params[0, 0] - stock_on_hand_and_in_transit), a_min=0)
