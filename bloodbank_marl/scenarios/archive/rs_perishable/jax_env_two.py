import jax
import chex
from typing import Tuple, Union, Optional, Dict, List
from flax import struct
import jax.numpy as jnp
from gymnax.environments import spaces
import numpy as np
import distrax
from bloodbank_marl.environments.marl_environment import (
    MarlEnvironment,
    EnvParams,
    EnvState,
    EnvInfo,
    EnvObs,
)
from bloodbank_marl.scenarios.rs_perishable.jax_env import (
    EnvState,
    EnvObs,
    EnvInfo,
    EnvParams,
    RSPerishableEnv,
)
from jax import lax

jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32

# NOTE: Differences to single agent version
# 1) state need to include full max_useful_life, because for issuing actions, all will be available
# and we need them to have the same dimensions.

n_products = 2
C = 1e10  # invalid substitution cost
max_useful_life = 3
# Based on Ensafian et al (2017)
substitution_cost_ratios = [
    # Unit RhD-, RhD+
    [
        0,
        C,
    ],  # RhD- pt
    [
        1 / 2,
        0,  # RhD+ pt
    ],
]
# These are from Ensafian et al (2017) - and similar to those in Meneses
product_probabilities = [0.16, 0.84]  # RhD-, RhD+

action_mask_per_request_type = np.array([[1, 0], [1, 1]])


# TODO: Recheck all defaults
@struct.dataclass
class EnvParams:
    poisson_demand_mean: float
    product_probabilities: chex.Array
    age_on_arrival_distribution_probs: chex.Array
    fixed_order_costs: float
    variable_order_costs: chex.Array
    shortage_costs: chex.Array
    wastage_costs: chex.Array
    holding_costs: chex.Array
    # For now, we assume that subsitution cost increases by 1/8
    # TODO: Check if this is what they meant (or whether, for example, if only one possible sub
    # then it is the wost and so should be 7/8)
    substitution_costs: chex.Array
    action_mask_per_request_type: chex.Array
    initial_weekday: int  # 0 Monday, 6, Sunday, -1 random on each reset
    max_expiry_pc_target: float
    min_service_level_pc_target: float
    min_exact_match_pc_target: float
    target_kpi_breach_penalty: float
    max_days_in_episode: int
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        cls,
        poisson_demand_mean: float = [
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
        + [0] * (max_useful_life - 1),  # All fresh
        fixed_order_costs: float = 225.0,
        variable_order_costs: List[float] = [650] * n_products,
        shortage_costs: List[float] = [3250] * n_products,
        wastage_costs: List[float] = [650] * n_products,
        holding_costs: List[float] = [130] * n_products,
        substitution_cost_ratios: List[List[float]] = substitution_cost_ratios,
        max_substitution_cost: float = 3250,
        action_mask_per_request_type: chex.Array = action_mask_per_request_type,
        initial_weekday: int = 0,  # Start on Monday morning; equiv to starting on Sunday evening before
        max_expiry_pc_target: float = 100.0,  # No limit by default
        min_service_level_pc_target: float = 0.0,  # No limit by default
        min_exact_match_pc_target: float = 0.0,
        target_kpi_breach_penalty: float = 0.0,  # No penalty for now
        max_days_in_episode: int = 365,
        max_steps_in_episode: int = 1e6,  # Much higher than expected
        gamma: float = 1.0,
    ):
        return cls(
            jnp.array(poisson_demand_mean),
            jnp.array(product_probabilities),
            jnp.array(age_on_arrival_distribution_probs),
            fixed_order_costs,
            jnp.array(variable_order_costs),
            jnp.array(shortage_costs),
            jnp.array(wastage_costs),
            jnp.array(holding_costs),
            jnp.array(substitution_cost_ratios) * max_substitution_cost,
            jnp.array(action_mask_per_request_type),
            initial_weekday,
            max_expiry_pc_target,
            min_service_level_pc_target,
            min_exact_match_pc_target,
            target_kpi_breach_penalty,
            max_days_in_episode,
            max_steps_in_episode,
            gamma,
        )


class RSPerishableTwoEnv(RSPerishableEnv):
    def __init__(
        self,
        agent_names=["replenishment", "issuing"],
        n_products: int = n_products,
        max_useful_life: int = max_useful_life,
        lead_time: int = 0,
        max_order_quantities: list = [50] * n_products,
        max_demand=100,
    ):
        self.n_products = n_products
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantities = jnp.array(max_order_quantities)
        self.max_demand = max_demand

        self.possible_agents = agent_names
        self.agent_ids = {agent_name: i for i, agent_name in enumerate(agent_names)}
        self.num_agents = len(agent_names)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    @property
    def name(self) -> str:
        """Environment name."""
        return "RSPerishableTwo"
