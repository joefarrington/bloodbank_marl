import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from typing import Tuple, Optional, Dict
import chex
from flax import struct
import numpyro

import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import time
import matplotlib.pyplot as plt
from typing import Optional, Callable, Dict, Any, Tuple
from functools import partial
import orbax
import wandb
import hydra
import omegaconf

# TODO Things we could add to this implementation of PPO
# - Advantage scaling
# - Reward scaling
# - Log the losses
# - Log infos/KPIs
# - Periodic deterministic evaluation (probably on a spearate environment in a log wrapper)
# - Bootstrapping when we get truncation


# Copied from viso_jax
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


# Avoid warnings by using standard int based on whether
# double precision is enabled or not
jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


class DeMoorPerishableGymnax(environment.Environment):
    def __init__(
        self,
        max_useful_life: int = 2,
        lead_time: int = 1,
        max_order_quantity: int = 10,
        issue_policy: str = "lifo",
    ):
        super().__init__()
        self.max_useful_life = max_useful_life
        self.lead_time = lead_time
        self.max_order_quantity = max_order_quantity
        self.issue_policy = issue_policy

    @property
    def default_params(self) -> EnvParams:
        return EnvParams.create_env_params()

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
            self.issue_policy == "fifo",
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
                # "discount": self.discount(state, params),
                # "cumulative_gamma": cumulative_gamma,
                # "demand": demand,
                # "shortage": shortage,
                # "holding": holding,
                # "expiries": expiries,
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
        return jnp.array([*state.in_transit, *state.stock])

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
        if params is None:
            params = self.default_params
        obs_len = self.max_useful_life + self.lead_time - 1
        low = jnp.array([0] * obs_len)
        high = jnp.array([self.max_order_quantity] * obs_len)
        return spaces.Box(
            low,
            high,
            (obs_len,),
            dtype=jnp_int,
        )

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

    @classmethod
    def calculate_kpis(cls, rollout_results: Dict) -> Dict[str, float]:
        """Calculate KPIs for each rollout, using the output of a rollout from RolloutWrapper"""
        service_level = (
            rollout_results["info"]["demand"] - rollout_results["info"]["shortage"]
        ).sum(axis=-1) / rollout_results["info"]["demand"].sum(axis=-1)
        wastage = rollout_results["info"]["expiries"].sum(axis=-1) / rollout_results[
            "action"
        ].sum(axis=(-1))
        holding_units = rollout_results["info"]["holding"].mean(axis=-1)
        return {
            "service_level_%": service_level * 100,
            "wastage_%": wastage * 100,
            "holding_units": holding_units,
        }


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    num_hidden_units: int
    num_hidden_layers: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(self.num_hidden_units)(x)
        actor_mean = activation(actor_mean)
        for i in range(1, self.num_hidden_layers):
            actor_mean = nn.Dense(self.num_hidden_units)(actor_mean)
            actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim)(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.num_hidden_units)(x)
        critic = activation(critic)
        for i in range(1, self.num_hidden_layers):
            critic = nn.Dense(self.num_hidden_units)(critic)
            critic = activation(critic)
        critic = nn.Dense(1)(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


@struct.dataclass
class HPConfig:
    LR: float
    GAMMA: float
    GAE_LAMBDA: float
    CLIP_EPS: float
    ENT_COEF: float
    VF_COEF: float
    MAX_GRAD_NORM: float


def make_train(fixed_config):
    fixed_config["NUM_UPDATES"] = (
        fixed_config["TOTAL_TIMESTEPS"]
        // fixed_config["NUM_STEPS"]
        // fixed_config["NUM_ENVS"]
    )
    fixed_config["MINIBATCH_SIZE"] = (
        fixed_config["NUM_ENVS"]
        * fixed_config["NUM_STEPS"]
        // fixed_config["NUM_MINIBATCHES"]
    )
    # env, env_params = gymnax.make(config["ENV_NAME"])
    env, env_params = DeMoorPerishableGymnax(), DeMoorPerishableGymnax().default_params
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(hp_config, rng):
        def linear_schedule(count):
            frac = (
                1.0
                - (
                    count
                    // (fixed_config["NUM_MINIBATCHES"] * fixed_config["UPDATE_EPOCHS"])
                )
                / fixed_config["NUM_UPDATES"]
            )
            return hp_config.LR * frac

        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).n,
            num_hidden_units=fixed_config["NUM_HIDDEN_UNITS"],
            num_hidden_layers=fixed_config["NUM_HIDDEN_LAYERS"],
            activation=fixed_config["ACTIVATION"],
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if fixed_config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(hp_config.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(hp_config.MAX_GRAD_NORM),
                optax.adam(hp_config.LR, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, fixed_config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, fixed_config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, fixed_config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + hp_config.GAMMA * next_value * (1 - done) - value
                    gae = (
                        delta
                        + hp_config.GAMMA * hp_config.GAE_LAMBDA * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-hp_config.CLIP_EPS, hp_config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - hp_config.CLIP_EPS,
                                1.0 + hp_config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + hp_config.VF_COEF * value_loss
                            - hp_config.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = (
                    fixed_config["MINIBATCH_SIZE"] * fixed_config["NUM_MINIBATCHES"]
                )
                assert (
                    batch_size == fixed_config["NUM_STEPS"] * fixed_config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [fixed_config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, fixed_config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)

        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, fixed_config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


class RolloutWrapper(object):
    def __init__(
        self,
        model_forward: Callable = None,
        env_id: str = "DeMoorPerishable",
        num_env_steps: Optional[int] = None,
        env_kwargs: Dict[str, Any] = {},
        env_params: Dict[str, Any] = {},
        num_burnin_steps: int = 0,
        return_info: bool = False,
    ):
        """Wrapper to define batch evaluation for policy parameters."""
        self.env_id = env_id
        # Define the RL environment & network forward function
        self.env, default_env_params = (
            DeMoorPerishableGymnax(),
            DeMoorPerishableGymnax().default_params,
        )

        if num_env_steps is None:
            self.num_env_steps = default_env_params.max_steps_in_episode
        else:
            self.num_env_steps = num_env_steps

        # Run a total of num_burnin_steps + num_env_steps
        # The burn-in steps are run first, and not included
        # in the reported outputs
        self.num_burnin_steps = num_burnin_steps

        # None of our environments have a fixed number of steps
        # so set to match desired number of steps
        env_params["max_steps_in_episode"] = self.num_env_steps + self.num_burnin_steps
        self.env_params = default_env_params.create_env_params(**env_params)
        self.model_forward = model_forward

        # If True, include info from each step in output
        self.return_info = return_info

    @partial(jax.jit, static_argnums=(0,))
    def population_rollout(
        self, rng_eval: chex.PRNGKey, policy_params: chex.Array
    ) -> Dict[str, chex.Array]:
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over rng & params
        pop_rollout = jax.vmap(self.batch_rollout, in_axes=(None, 0))
        return pop_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0,))
    def batch_rollout(
        self, rng_eval: chex.PRNGKey, policy_params: chex.Array
    ) -> Dict[str, chex.Array]:
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout, in_axes=(0, None))
        return batch_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0,))
    def single_rollout(
        self, rng_input: chex.PRNGKey, policy_params: chex.Array
    ) -> Dict[str, chex.Array]:
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def policy_step(state_input, tmp):
            """lax.scan compatible step transition in jax env."""
            (
                obs,
                state,
                policy_params,
                rng,
                discounted_cum_reward,
                valid_mask,
            ) = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            if self.model_forward is not None:
                action = self.model_forward(policy_params, obs, rng_net)
            else:
                action = self.env.action_space(self.env_params).sample(rng_net)
            next_obs, next_state, reward, done, info = self.env.step(
                rng_step, state, action, self.env_params
            )

            new_discounted_cum_reward = discounted_cum_reward + jnp.where(
                state.step >= self.num_burnin_steps,
                reward
                * valid_mask
                * (
                    self.env.cumulative_gamma(state, self.env_params)
                    / self.env_params.gamma**self.num_burnin_steps
                ),
                0,
            )

            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_obs,
                next_state,
                policy_params,
                rng,
                new_discounted_cum_reward,
                new_valid_mask,
            ]

            if self.return_info:
                y = [
                    obs,
                    action,
                    reward,
                    next_obs,
                    done,
                    info,
                ]
            else:
                y = [obs, action, reward, next_obs, done]

            return carry, y

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                policy_params,
                rng_episode,
                jnp.array([0.0]),
                jnp.array([1.0]),
            ],
            (),
            self.num_env_steps + self.num_burnin_steps,
        )

        output = {}
        start_idx = self.num_burnin_steps
        stop_idx = self.num_burnin_steps + self.num_env_steps
        if self.return_info:
            (
                obs,
                action,
                reward,
                next_obs,
                done,
                info,
            ) = scan_out
            output["info"] = {k: v[start_idx:stop_idx] for k, v in info.items()}
            output["info"]["cumulative_gamma"] = output["info"]["cumulative_gamma"] / (
                self.env_params.gamma**self.num_burnin_steps
            )  # Discounting start from end of burnin period
        else:
            obs, action, reward, next_obs, done = scan_out

        # Extract the discounted sum of rewards accumulated by agent in episode rollout
        cum_return = carry_out[-2]

        output["obs"] = obs[start_idx:stop_idx]
        output["action"] = action[start_idx:stop_idx]
        output["reward"] = reward[start_idx:stop_idx]
        output["next_obs"] = next_obs[start_idx:stop_idx]
        output["done"] = done[start_idx:stop_idx]
        output["cum_return"] = cum_return

        return output

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape


def deterministic_fwd(params, obs, rng, network):
    pi, _ = network.apply(params, obs)
    return pi.mode()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    # Resolve these to dicts because make_train adds extra elements
    # and we want to use hp_config as input to the dataclass
    fixed_config = omegaconf.OmegaConf.to_container(cfg.fixed_config, resolve=True)
    hp_config = HPConfig(
        **omegaconf.OmegaConf.to_container(cfg.hp_config, resolve=True)
    )
    train_vjit = jax.jit(jax.vmap(make_train(fixed_config), in_axes=(None, 0)))
    rng_train, rng_eval = jax.random.split(jax.random.PRNGKey(cfg.seed), 2)
    outs = train_vjit(hp_config, jax.random.split(rng_train, cfg.n_seeds))

    # Loop over the output to log
    returns = np.array(
        outs["metrics"]["returned_episode_returns"].mean(-1).reshape(cfg.n_seeds, -1)
    )
    for i in range(returns.shape[1]):
        log_to_wandb = {
            "mean_returned_episode_returns": returns[:, i].mean(),
        }
        # Note that the times on these logs will be off, but that's not really important
        wandb.log(log_to_wandb)

    ### Evaluate deterministic policy on new rollouts ###
    env, env_params = DeMoorPerishableGymnax(), DeMoorPerishableGymnax().default_params
    network = ActorCritic(
        env.action_space(env_params).n,
        num_hidden_units=fixed_config["NUM_HIDDEN_UNITS"],
        num_hidden_layers=fixed_config["NUM_HIDDEN_LAYERS"],
        activation=fixed_config["ACTIVATION"],
    )
    network_forward = partial(deterministic_fwd, network=network)
    rw = RolloutWrapper(
        model_forward=network_forward,
        env_id="DeMoorPerishable",
        num_env_steps=365,
        env_kwargs={},
        env_params={},
        num_burnin_steps=100,
        return_info=False,
    )
    policy_params = outs["runner_state"][0].params
    eval_out = rw.population_rollout(
        jax.random.split(rng_eval, cfg.n_eval_episodes), policy_params
    )
    print(eval_out["cum_return"].shape)
    wandb.log({"mean_eval_return": eval_out["cum_return"].mean()})


if __name__ == "__main__":
    main()
