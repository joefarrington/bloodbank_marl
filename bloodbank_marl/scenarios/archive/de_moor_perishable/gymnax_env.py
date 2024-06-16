# Originally from https://github.com/joefarrington/viso_jax/blob/main/viso_jax/scenarios/de_moor_perishable/environment.py
# Rewrote the method for calculating the KPIs
# Added an EnvObs class, and action mask
# Added ameneded step function to use EnvObs
# Added end of warmup reset function
# Added calculate target KPI penalty

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import Tuple, Union, Optional, List, Dict
import chex
from flax import struct
import numpyro
from functools import partial


@struct.dataclass
class EnvState:
    in_transit: chex.Array
    stock: chex.Array
    step: int


@struct.dataclass
class EnvParams:
    max_demand: int
    demand_gamma_alpha: float
    demand_gamma_beta: float
    cost_components: chex.Array
    max_steps_in_episode: int
    gamma: float

    @classmethod
    def create_env_params(
        # Default env params are for m=2, experiment 1
        cls,
        max_demand: int = 100,
        demand_gamma_mean: float = 4.0,
        demand_gamma_cov: float = 0.5,
        variable_order_cost: float = 3.0,
        shortage_cost: float = 5.0,
        wastage_cost: float = 7.0,
        holding_cost: float = 1.0,
        max_steps_in_episode: int = 365,
        gamma: float = 0.99,
    ):
        demand_gamma_alpha = 1 / (demand_gamma_cov**2)
        demand_gamma_beta = 1 / (demand_gamma_mean * demand_gamma_cov**2)
        cost_components = jnp.array(
            [
                variable_order_cost,
                shortage_cost,
                wastage_cost,
                holding_cost,
            ]
        )
        return cls(
            max_demand,
            demand_gamma_alpha,
            demand_gamma_beta,
            cost_components,
            max_steps_in_episode,
            gamma,
        )


@struct.dataclass
class EnvObs:
    stock: chex.Array
    in_transit: chex.Array
    action_mask: chex.Array

    @property
    def obs(self):
        batch_dims = self.in_transit.shape[:-1]
        return jnp.hstack(
            [
                self.in_transit.reshape(batch_dims + (-1,)),
                self.stock.reshape(batch_dims + (-1,)),
            ]
        )


# Avoid warnings by using standard int based on whether
# double precision is enabled or not
jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


class DeMoorPerishableGymnax(environment.Environment):
    def __init__(
        self,
        max_useful_life: int = 2,
        lead_time: int = 1,
        max_order_quantity: int = 10,
        issuing_policy: str = "fifo",
    ):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantity = max_order_quantity
        self.issuing_policy = issuing_policy

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: Union[int, float],
        params: Optional[EnvParams] = None,
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Performs step transitions in the environment."""
        # Use default env parameters if no others specified
        if params is None:
            params = self.default_params
        key, key_reset = jax.random.split(key)
        obs_st, state_st, reward, done, info = self.step_env(key, state, action, params)
        obs_re, state_re = self.reset_env(key_reset, params)
        # Auto-reset environment based on termination
        state = jax.tree_map(
            lambda x, y: jax.lax.select(done, x, y), state_re, state_st
        )
        obs = jax.tree_map(lambda x, y: jax.lax.select(done, x, y), obs_re, obs_st)
        return obs, state, reward, done, info

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict]:
        """Performs step transitions in the environment."""
        prev_terminal = self.is_terminal(state, params)
        cumulative_gamma = self.cumulative_gamma(state, params)

        # Add ordered unit to in_transit
        in_transit = jnp.hstack([action, state.in_transit])

        # Generate demand
        demand_dist = numpyro.distributions.Gamma(
            concentration=params.demand_gamma_alpha, rate=params.demand_gamma_beta
        )
        demand = (
            jnp.round(demand_dist.sample(key=key))
            .clip(0, params.max_demand)  # Truncate at max demand
            .astype(jnp_int)
        )

        # Meet demand
        stock_after_issue = jax.lax.cond(
            self.issuing_policy == "fifo",
            self._issue_fifo,
            self._issue_lifo,
            state.stock,
            demand,
        )

        # Compute variables required to calculate the
        variable_order = jnp.array(action)
        shortage = jnp.max(jnp.array([demand - jnp.sum(state.stock), 0]))
        expiries = stock_after_issue[-1]
        holding = jnp.sum(stock_after_issue[: self.max_useful_life - 1])

        # Same order as params.cost_components
        transition_function_reward_output = jnp.array(
            [variable_order, shortage, expiries, holding]
        )

        # Calculate reward
        reward = self._calculate_single_step_reward(
            state, action, transition_function_reward_output, params
        )

        # Update the state - age, stock, receive order placed at step t-(L-1)
        # This order is assumed to arrive just prior to the start of the next
        # period, and so is included in the updated state but no holding costs
        # are charged on it
        closing_stock = jnp.hstack(
            [in_transit[-1], stock_after_issue[: self.max_useful_life - 1]]
        )
        closing_in_transit = in_transit[0 : self.lead_time - 1]
        state = EnvState(closing_in_transit, closing_stock, state.step + 1)
        done = self.is_terminal(state, params)

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward,
            done,
            {
                "discount": self.discount(state, params),
                "cumulative_gamma": cumulative_gamma,
                "day_counter": 1,
                "order": action,
                "demand": demand,
                "shortage": shortage,
                "holding": holding,
                "expiries": expiries,
            },
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Performs resetting of environment."""
        # Always start with no stock and nothing in transit
        state = EnvState(
            in_transit=jnp.zeros(self.lead_time - 1, dtype=jnp_int),
            stock=jnp.zeros(self.max_useful_life, dtype=jnp_int),
            step=0,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState) -> chex.Array:
        """Applies observation function to state."""
        return EnvObs(
            stock=state.stock,
            in_transit=state.in_transit,
            action_mask=jnp.ones((self.max_order_quantity + 1,), dtype=jnp_int),
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Check number of steps in episode termination condition
        done_steps = state.step >= params.max_steps_in_episode
        return done_steps

    def cumulative_gamma(self, state: EnvState, params: EnvParams) -> float:
        """Return cumulative discount factor"""
        return params.gamma**state.step

    def _calculate_single_step_reward(
        self,
        state: EnvState,
        action: int,
        transition_function_reward_output: chex.Array,
        params: EnvParams,
    ) -> int:
        """Calculate reward for a single step transition"""
        cost = jnp.dot(transition_function_reward_output, params.cost_components)
        reward = -1 * cost
        return reward

    def _issue_fifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using FIFO policy"""
        _, remaining_stock = jax.lax.scan(
            self._issue_one_step, demand, opening_stock, reverse=True
        )
        return remaining_stock

    def _issue_lifo(self, opening_stock: chex.Array, demand: int) -> chex.Array:
        """Issue stock using LIFO policy"""
        _, remaining_stock = jax.lax.scan(self._issue_one_step, demand, opening_stock)
        return remaining_stock

    def _issue_one_step(
        self, remaining_demand: int, stock_element: int
    ) -> Tuple[int, int]:
        """Fill demand with stock of one age, representing one element in the state"""
        remaining_stock = (stock_element - remaining_demand).clip(0)
        remaining_demand = (remaining_demand - stock_element).clip(0)
        return remaining_demand, remaining_stock

    @property
    def name(self) -> str:
        """Environment name."""
        return "DeMoorPerishable"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.max_order_quantity + 1

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        """Action space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Discrete(self.max_order_quantity + 1)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        # [O, X], in_transit and in_stock, stock ages left to right
        raise NotImplementedError

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        if params is None:
            params = self.default_params
        return spaces.Dict(
            {
                "in_transit": spaces.Box(
                    0, self.max_order_quantity, (self.lead_time - 1,), jnp_int
                ),
                "stock": spaces.Box(
                    0, self.max_order_quantity, (self.max_useful_life,), jnp_int
                ),
                "step": spaces.Discrete(params.max_steps_in_episode),
            }
        )

    @property
    def empty_infos(self) -> Dict[str, Union[chex.Array, float, int]]:
        return {
            "cumulative_gamma": 1.0,
            "discount": 0.0,
            "day_counter": 0,
            "demand": 0,
            "expiries": 0,
            "holding": 0,
            "order": 0,
            "shortage": 0,
        }

    @classmethod
    def calculate_target_kpi_penalty(
        cls, kpis: Dict[str, Union[chex.Array, float]], params
    ):
        # No target KPIs for this environment
        return 0

    def calculate_kpis(
        self, cum_info: Dict[str, Union[chex.Array, float]]
    ) -> Dict[str, Union[chex.Array, float]]:
        """Calculate KPIs based on the info recorded by the replenishment agent, with id 0"""
        return {
            "service_level_%": (
                jnp.sum(cum_info["demand"] - cum_info["shortage"]) * 100
            )
            / jnp.sum(cum_info["demand"]),
            "wastage_%": (jnp.sum(cum_info["expiries"]) * 100)
            / jnp.sum(cum_info["order"]),
            "mean_holding": jnp.sum(cum_info["holding"]) / cum_info["day_counter"],
            "day_count": cum_info["day_counter"],
        }

    def end_of_warmup_reset(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ):
        """Run at end of warmup period to partially reset State"""
        _, state_reset = self.reset(key, params)
        # We want to keep the stock on hand and in transit, but reset everything else
        return state_reset.replace(
            stock=state.stock, in_transit=state.in_transit, step=0
        )
