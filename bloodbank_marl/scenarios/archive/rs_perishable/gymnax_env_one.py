# A single agent environment in Gymnax, with the functionality to add in a issuing policy as an argument
# Not true multiagent, because we don't get states for other agents, but we can use this to test the issuing policy and
# the rep policy at the same time.
# Currently a lot quicker than the multiagent env, so much better for testing "fixed" issuing policies like exact match
# and fixed priority.


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
from bloodbank_marl.scenarios.rs_perishable.gymnax_env import (
    EnvState,
    EnvObs,
    DemandInfo,
    IssueObs,
    RSPerishableGymnax,
)

n_products = 1
C = 1e10  # invalid substitution cost
max_useful_life = 3
substitution_cost_ratios = [[0]]
product_probabilities = [1.0]  


@struct.dataclass
class EnvParams:
    poisson_demand_mean: chex.Array
    product_probabilities: chex.Array  # NOTE: For now, assume same for all days of week
    age_on_arrival_distribution_probs: (
        chex.Array
    )  # NOTE: For now, assume same for all days of week
    fixed_order_cost: float
    variable_order_costs: chex.Array
    shortage_costs: chex.Array
    wastage_costs: chex.Array
    holding_costs: chex.Array
    # For now, we assume that subsitution cost increases by 1/8
    # TODO: Check if this is what they meant (or whether, for example, if only one possible sub
    # then it is the wost and so should be 7/8)
    substitution_costs: chex.Array
    # TODO: Add in functions to apply penalty for breaching targets
    initial_weekday: int  # 0 Monday, 6, Sunday, -1 random on each reset
    max_expiry_pc_target: float
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
        fixed_order_cost: float = 225,
        variable_order_costs: List[float] = [650] * n_products,
        shortage_costs: List[float] = [3250] * n_products,
        wastage_costs: List[float] = [650] * n_products,
        holding_costs: List[float] = [130] * n_products,
        substitution_cost_ratios: List[List[float]] = substitution_cost_ratios,
        max_substitution_cost: float = 3250,  # Max substitution equal to shortage costs, like Meneses
        initial_weekday: int = -1,  # Start on random weekday
        max_expiry_pc_target: float = 100.0,  # Effectively no limit by default
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
            fixed_order_cost,
            jnp.array(variable_order_costs),
            jnp.array(shortage_costs),
            jnp.array(wastage_costs),
            jnp.array(holding_costs),
            jnp.array(substitution_cost_ratios) * max_substitution_cost,
            initial_weekday,
            max_expiry_pc_target,
            min_service_level_pc_target,
            min_exact_match_pc_target,
            target_kpi_breach_penalty,
            max_steps_in_episode,
            gamma,
        )


class RSPerishableOneGymnax(RSPerishableGymnax):
    def __init__(
        self,
        n_products: int = n_products,
        max_useful_life: int = 3,
        lead_time: int = 0,
        max_order_quantities: list = [50] * n_products,  # TODO: Check older work
        max_demand: int = 100,  # TODO: Check older work
        issuing_policy=None,  # TODO: For now, set as None
    ):
        super().__init__()
        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantities = jnp.array(max_order_quantities)
        self.max_demand = max_demand
        self._issuing_policy = issuing_policy

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    @property
    def name(self) -> str:
        """Environment name."""
        return "RSPerishableOneGymnax"
