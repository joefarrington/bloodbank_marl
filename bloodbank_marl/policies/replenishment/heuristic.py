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
        self.env_params = default_env_params.create_env_params(**env_params)
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
        if self.env_name in [
            "MenesesPerishable",
            "MenesesPerishableGymnax",
            "RSPerishable",
            "RSPerishableGymnax",
        ]:
            return ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
        elif self.env_name == "RSPerishableFourGymnax":
            return ["O", "A", "B", "AB"]
        elif self.env_name in ["RSPerishableTwoGymnax", "RSPerishable"]:
            return ["RhD-", "RhD+"]
        else:
            return []

    def _get_apply_method(self) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        if self.env_name == "DeMoorPerishable":
            return de_moor_perishable_S_policy
        elif self.env_name == "MenesesPerishable" or "MenesesPerishableGymnax":
            return meneses_perishable_S_policy
        elif (
            self.env_name in ["RSPerishable", "RSPerishableTwo", "RSPerishableGymnax", "RSPerishableFourGymnax", "RSPerishableTwoGymnax", "RSPerishableOneGymnax"]
        ):
            return rs_perishable_S_policy
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


# NOTE: We probably want to version of this with params per weekday
def rs_perishable_S_policy(policy_params, obs, rng):
    """S policy for scenario based on R&S, but with added blood groups"""
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


# For use in with pretraining in large state spaces where we need to collect
# samples instead of enumerating the possible states
class SRepPolicyExplore(SRepPolicy):
    def __init__(
        self,
        env_name=None,
        env_kwargs={},
        env_params={},
        fixed_policy_params=None,  # Enables us to fix policy params for convenience
        params_min=None,
        params_max=None,
        epsilon=0.05,
    ):

        self.env_name = env_name
        self.env_kwargs = env_kwargs
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.env_params = default_env_params.create_env_params(**env_params)
        self.obs, _ = env.reset(jax.random.PRNGKey(0), self.env_params)

        # Set up parameters
        self._setup_param_properties()

        # For now, force sample limit on all S params
        self.params_min = params_min
        self.params_max = params_max
        self.epsilon = epsilon

        self.param_shape = self

        # Set up apply method
        self._apply = self._get_apply_method()

        self.apply = self.apply_with_exploration_noise

    def apply_with_exploration_noise(self, policy_params, obs, rng):
        """Apply the policy to the observation, adding noise to the action"""
        rng, _rng = jax.random.split(rng)

        # Will we replace each S parameter?
        rng, _rng = jax.random.split(rng)
        sample_exploration_noise = jax.random.uniform(
            _rng, shape=self.params_shape, minval=0, maxval=1
        )

        # What will we replace each S parameter with?
        rng, _rng = jax.random.split(rng)
        sample_replacement_S = jax.random.poisson(
            _rng, shape=policy_params.shape, lam=policy_params
        )
        S = jnp.where(
            sample_exploration_noise < self.epsilon, sample_replacement_S, policy_params
        )

        tr_action = self._apply(S, obs, _rng)
        action = self._postprocess_action(obs, tr_action)
        return action


class sSRepPolicy(SRepPolicy):
    # (s,S) policy with a pair of parameters for each product

    def _get_param_col_names(self) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["s", "S"]

    def _get_param_row_names(self) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if self.env_name in [
            "RSPerishableGymnax", "RSPerishable"
        ]:
            return ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
        elif self.env_name == "RSPerishableFourGymnax":
            return ["O", "A", "B", "AB"]
        elif self.env_name in ["RSPerishableTwo", "RSPerishableTwoGymnax"]:
            return ["RhD-", "RhD+"]
        else:
            return []

    def _get_apply_method(self) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        if self.env_name == "RSPerishableGymnax":
            return rs_perishable_sS_policy
        else:
            raise NotImplementedError(
                f"No (s,S) policy defined for Environment ID {self.env_name}"
            )


def rs_perishable_sS_policy(policy_params, obs, rng):
    """S policy for scenario based on R&S, but with added blood groups"""
    stock_on_hand_and_in_transit = obs.stock.sum(
        axis=-1, keepdims=True
    ) + obs.in_transit.sum(axis=-1, keepdims=True)
    # Clip because we can't have a negative order
    # Squeeze because action should be a 1D array
    order = jnp.clip(
        jnp.where(
            stock_on_hand_and_in_transit <= policy_params[:, 0].reshape(-1, 1),
            (policy_params[:, 1].reshape(-1, 1) - stock_on_hand_and_in_transit),
            0,
        ),
        a_min=0,
    ).squeeze()
    # To enforce that s<S, order nothing if any s >= S.
    return jax.lax.select(
        jnp.all(policy_params[:, 0] < policy_params[:, 1]),
        order,
        jnp.zeros_like(order),
    )


class SDayOfWeekRepPolicy(SRepPolicy):
    # S policy with a pair of parameters for each product for each day of week

    def _get_param_col_names(self) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return [f"S_{w}" for w in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]

    def _get_param_row_names(self) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if self.env_name in [
            "RSPerishable", "RSPerishableGymnax",
        ]:
            return ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
        elif self.env_name == "RSPerishableFourGymnax":
            return ["O", "A", "B", "AB"]
        elif self.env_name in ["RSPerishableTwo", "RSPerishableTwoGymnax"]:
            return ["RhD-", "RhD+"]
        elif self.env_name == "RSPerishableOneGymnax":
            return ["Plt"]
        else:
            return []

    def _get_apply_method(self) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        if self.env_name in [
            "RSPerishableGymnax",
            "RSPerishableFourGymnax",
            "RSPerishableTwoGymnax",
            "RSPerishableOneGymnax",
        ]:
            return rs_perishable_S_day_of_week_policy
        else:
            raise NotImplementedError(
                f"No (S) day of week policy defined for Environment ID {self.env_name}"
            )


def rs_perishable_S_day_of_week_policy(policy_params, obs, rng):
    """S policy for scenario based on R&S, but with added blood groups"""
    stock_on_hand_and_in_transit = obs.stock.sum(
        axis=-1, keepdims=True
    ) + obs.in_transit.sum(axis=-1, keepdims=True)
    S = policy_params[:, obs.weekday].reshape(-1, 1)
    return jnp.clip(S - stock_on_hand_and_in_transit, a_min=0).squeeze()


class sSDayOfWeekRepPolicy(SRepPolicy):
    # S policy with a pair of parameters for each product for each day of week

    def _get_param_col_names(self) -> List[str]:
        """Get the column names for the policy parameters - these are the different types
        of parameters e.g. target stock level or reorder point"""
        return ["s", "S"]

    def _get_param_row_names(self) -> List[str]:
        """Get the row names for the policy parameters - these are the names of the different levels of a
        given paramter, e.g. for different days of the week or different products"""
        if self.env_name in [
            "RSPerishableOneGymnax",
            "MirjaliliPerishablePlateletGymnax",
        ]:
            return [f"S_{w}" for w in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]]
        else:
            return []

    def _get_apply_method(self) -> callable:
        """Get the forward method for the policy - this is the function that returns the action"""
        if self.env_name == "RSPerishableOneGymnax":
            return rs_perishable_sS_day_of_week_policy
        elif self.env_name == "MirjaliliPerishablePlateletGymnax":
            return mirjalili_perishable_sS_day_of_week_policy
        else:
            raise NotImplementedError(
                f"No (S) day of week policy defined for Environment ID {self.env_name}"
            )


def rs_perishable_sS_day_of_week_policy(policy_params, obs, rng):
    """sS policy for scenario based on R&S, with single product and weekdays"""
    stock_on_hand_and_in_transit = obs.stock.sum(
        axis=-1, keepdims=True
    ) + obs.in_transit.sum(axis=-1, keepdims=True)
    S = policy_params[obs.weekday, 1]
    s = policy_params[obs.weekday, 0]
    order = jnp.clip(
        jnp.where(
            stock_on_hand_and_in_transit <= s, S - stock_on_hand_and_in_transit, 0
        ).squeeze(),
        a_min=0,
    )
    return jax.lax.select(jnp.all(policy_params[:, 0] < policy_params[:, 1]), order, jnp.zeros_like(order))


def mirjalili_perishable_sS_day_of_week_policy(policy_params, obs, rng):
    stock_on_hand_and_in_transit = obs.stock.sum(axis=-1, keepdims=True)
    S = policy_params[obs.weekday, 1]
    s = policy_params[obs.weekday, 0]
    order = jnp.clip(
        jnp.where(
            stock_on_hand_and_in_transit <= s, S - stock_on_hand_and_in_transit, 0
        ).squeeze(),
        a_min=0,
    )
    return jax.lax.select(jnp.all(policy_params[:, 0] < policy_params[:, 1]), order, jnp.zeros_like(order))
