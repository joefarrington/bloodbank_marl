import gymnax
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
import chex
from typing import Tuple, Union, Optional, List, Dict
from functools import partial
from flax import struct
from jax import lax
import distrax
import matplotlib.pyplot as plt
import numpy as np
from bloodbank_marl.scenarios.simple_two_product.gymnax_env_try_issue_too import (
    SimpleTwoProductPerishableIncIssueGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_try_issue_too import (
    RSPerishableIncIssueGymnax,
)

# TODO: Fill out missing methods (e.g. for action/obs/state spaces etc) if we end up using this for final results

# TODO: Decide where to specify n_products and how to get it into default params

n_products = 4
M = 1e10  # invalid substitution cost
max_useful_life = 3

# Base on Yu Suen et al (2023), but consider only RhD+ version
substitution_cost_ratios = [
    # Unit O, A, B, AB
    [0, 3, 1, M],  # O+ patient
    [M, 0, 1, 2],  # A+ patient
    [M, 2, 0, 1],  # B+ patien
    [M, 1, 2, 0],  # AB+ patient
]

# Preferences for issuing policy based on the above
# [0, 4, 2, -1, -1, -1, -1, -1] # O- patient
# [1, 0, 5, 4, 3, 2, -1, -1] # O+ patient
# [2, 6, 4, 0, 3, 7, -1, -1] # A- patient
# [3, 2, 7, 6, 5, 4, -1, -1] # A+ patient
# [4, 6, 2, 0, 5, 7, -1, -1] # B- patient
# [5,4, 7, 6, 3, 2, -1, -1] #B+ patient
# [6, 2, 4, 0, 7, 3, -1, -1] # AB- patient
# [7, 6, 3, 2, 5, 4, -1, -1] # AB+ patient

# These are from Ensafian et al (2017) - and similar to those in Meneses
# But, just take RhD+ and then divide by total % so they add to 1
product_probabilities = [
    0.45,  # O+
    0.40,  # A
    0.11,  # B
    0.04,  # AB
]

action_mask_per_request_type = jnp.where(
    jnp.array(substitution_cost_ratios) > n_products, 0, 1
)


@struct.dataclass
class EnvState:
    stock: chex.Array
    in_transit: chex.Array
    weekday: int
    step: int  # For the purposes of this env, a step is a day
    issue_policy_params: Optional[Dict] = None


@struct.dataclass
class EnvParams:
    poisson_demand_mean: chex.Array
    product_probabilities: chex.Array
    age_on_arrival_distribution_probs: chex.Array
    fixed_order_cost: float
    variable_order_costs: chex.Array
    shortage_costs: chex.Array
    wastage_costs: chex.Array
    holding_costs: chex.Array
    substitution_costs: chex.Array
    action_mask_per_request_type: chex.Array
    initial_weekday: int
    max_wastage_pc_target: float
    min_service_level_pc_target: float
    min_exact_match_pc_target: float
    target_kpi_breach_penalty: float
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        cls,
        poisson_demand_mean: List[float] = [
            37.5,
            37.3,
            39.2,
            37.8,
            40.5,
            27.2,
            28.4,
        ],
        product_probabilities: List[float] = product_probabilities,
        age_on_arrival_distribution_probs: List[float] = [1]
        + [0] * (max_useful_life - 1),
        fixed_order_cost: float = 0,
        variable_order_costs: List[float] = [650.0] * n_products,
        shortage_costs: List[float] = [3250.0] * n_products,
        wastage_costs: List[float] = [650] * n_products,
        holding_costs: List[float] = [130] * n_products,
        substitution_cost_ratios: List[List[float]] = substitution_cost_ratios,
        max_substitution_cost: float = 3250.0,
        action_mask_per_request_type: List[int] = action_mask_per_request_type,
        initial_weekday: int = -1,
        max_wastage_pc_target: float = 100.0,  # Effectively no limit by default
        min_service_level_pc_target: float = 0.0,  # Effectively no limit by default
        min_exact_match_pc_target: float = 0.0,  # Effectively no limit by default
        target_kpi_breach_penalty: float = 0.0,  # Set penalty to 0 for now, was having issue with this
        max_steps_in_episode: int = 365,  # By default, we run for a year
        gamma: float = 1.0,
    ):
        return cls(
            jnp.array(poisson_demand_mean),
            jnp.array(product_probabilities),
            jnp.array(age_on_arrival_distribution_probs),
            jnp.array(fixed_order_cost),
            jnp.array(variable_order_costs),
            jnp.array(shortage_costs),
            jnp.array(wastage_costs),
            jnp.array(holding_costs),
            jnp.array(substitution_cost_ratios) * (max_substitution_cost / n_products),
            jnp.array(action_mask_per_request_type),
            initial_weekday,
            max_wastage_pc_target,
            min_service_level_pc_target,
            min_exact_match_pc_target,
            target_kpi_breach_penalty,
            max_steps_in_episode,
            gamma,
        )


@struct.dataclass
class EnvObs:
    stock: chex.Array
    in_transit: chex.Array
    weekday: chex.Array
    action_mask: chex.Array

    @property
    def obs(self):
        batch_dims = self.in_transit.shape[:-2]
        return jnp.hstack(
            [
                self.one_hot_day_of_week().reshape(batch_dims + (-1,)),
                self.in_transit.reshape(batch_dims + (-1,)),
                self.stock.reshape(batch_dims + (-1,)),
            ]
        )

    @property
    def obs_total_per_product(self):
        batch_dims = self.in_transit.shape[:-2]
        return self.total_per_product().reshape(batch_dims + (-1,))

    @property
    def obs_total_per_product_and_weekday(self):
        batch_dims = self.in_transit.shape[:-2]
        return jnp.hstack(
            [
                self.one_hot_day_of_week().reshape(batch_dims + (-1,)),
                self.total_per_product().reshape(batch_dims + (-1,)),
            ]
        )

    def one_hot_day_of_week(self):
        return jax.nn.one_hot(self.weekday, 7)

    def total_per_product(self):
        return self.stock.sum(axis=-1) + self.in_transit.sum(axis=-1)


@struct.dataclass
class DemandInfo:
    total_demand: int
    remaining_demand: int
    shortages: int
    allocations: int
    remaining_stock: chex.Array
    in_transit: chex.Array
    weekday: int
    request_type_samples: chex.Array
    request_time_samples: chex.Array
    key: jax.random.PRNGKey
    issue_policy_params: Optional[Dict]
    action_mask_per_request_type: chex.Array


@struct.dataclass
class IssueObs:
    stock: chex.Array
    in_transit: chex.Array
    request_time: chex.Array
    request_type: chex.Array
    weekday: chex.Array
    action_mask: chex.Array

    @property
    def obs(self):
        batch_dims = self.in_transit.shape[:-2]
        return jnp.hstack(
            [
                self.one_hot_day_of_week().reshape(batch_dims + (-1,)),
                self.request_time.reshape(batch_dims + (-1,)),
                self.request_type.reshape(batch_dims + (-1,)),
                self.in_transit.reshape(batch_dims + (-1,)),
                self.stock.reshape(batch_dims + (-1,)),
            ]
        )

    def one_hot_day_of_week(self):
        return jax.nn.one_hot(self.weekday, 7)


class RSPerishableFourIncIssueGymnax(RSPerishableIncIssueGymnax):
    """Jittable abstract base class for all gymnax Environments."""

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    @property
    def name(self) -> str:
        """Environment name."""
        return "RSPerishableFourIncIssueGymnax"
