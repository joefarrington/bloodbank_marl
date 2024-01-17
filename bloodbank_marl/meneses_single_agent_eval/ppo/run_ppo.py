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
import time
import matplotlib.pyplot as plt
from typing import Optional, Callable, Dict, Any, Tuple
from functools import partial
import orbax
import wandb
import hydra
import omegaconf
from bloodbank_marl.utils.gymnax_fitness import make
from bloodbank_marl.scenarios.meneses_perishable.gymnax_env import EnvObs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# TODO: Need to think carefully about logging if we are vmapping over some config inputs
# Which this is sort of designed to handle (as as per the HPConfig class)

# NOTE: Eval currently NOT deterministic because getting odd results


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: EnvObs
    info: jnp.ndarray


@struct.dataclass
class HPConfig:
    lr: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float


import jax
import jax.numpy as jnp
import chex
import numpy as np
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0,))
    def reset(
        self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0, 0, 0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: environment.EnvState,
        action: Union[int, float],
        params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info = (
            {}
        )  # We don't need all the info from the environment for training rollouts
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info


def make_train(fixed_config):
    fixed_config["NUM_UPDATES"] = int(
        fixed_config["total_timesteps"]
        // fixed_config["num_steps"]
        // fixed_config["num_envs"]
    )
    fixed_config["MINIBATCH_SIZE"] = int(
        fixed_config["num_envs"]
        * fixed_config["num_steps"]
        // fixed_config["num_minibatches"]
    )
    env, env_params = make(
        fixed_config["environment"]["env_name"],
        **fixed_config["environment"]["env_kwargs"],
    )
    env = LogWrapper(env)

    def train(hp_config, rng):
        def linear_schedule(count):
            frac = (
                1.0
                - (
                    count
                    // (fixed_config["num_minibatches"] * fixed_config["update_epochs"])
                )
                / fixed_config["NUM_UPDATES"]
            )
            return hp_config.lr * frac

        policy = hydra.utils.instantiate(fixed_config["policies"]["replenishment"])
        rng, _rng = jax.random.split(rng)
        policy_params = policy.get_initial_params(_rng)

        if fixed_config["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(hp_config.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(hp_config.max_grad_norm),
                optax.adam(hp_config.LR, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=policy.apply,
            params=policy_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, fixed_config["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                action, tr_action, log_prob, value = policy.apply_for_training(
                    train_state.params, last_obs, _rng
                )
                # pi, value = network.apply(train_state.params, last_obs)
                # action = pi.sample(seed=_rng)
                # log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, fixed_config["num_envs"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                transition = Transition(
                    done, tr_action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, fixed_config["num_steps"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            # _, last_val = network.apply(train_state.params, last_obs)
            _, last_val = policy.model.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + hp_config.gamma * next_value * (1 - done) - value
                    gae = (
                        delta
                        + hp_config.gamma * hp_config.gae_lambda * (1 - done) * gae
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

                        log_prob, value, entropy = policy.apply_for_loss_fn(
                            params, traj_batch.obs, traj_batch.action
                        )
                        # pi, value = network.apply(params, traj_batch.obs)
                        # log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-hp_config.clip_eps, hp_config.clip_eps)
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
                                1.0 - hp_config.clip_eps,
                                1.0 + hp_config.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        # entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + hp_config.vf_coef * value_loss
                            - hp_config.ent_coef * entropy
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
                    fixed_config["MINIBATCH_SIZE"] * fixed_config["num_minibatches"]
                )
                assert (
                    batch_size == fixed_config["num_steps"] * fixed_config["num_envs"]
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
                        x,
                        [
                            fixed_config["num_minibatches"],
                            fixed_config["MINIBATCH_SIZE"],
                        ]
                        + list(x.shape[1:]),
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
                _update_epoch, update_state, None, fixed_config["update_epochs"]
            )
            train_state = update_state[0]
            metric = {
                "loss": loss_info,
                "returned_episode_returns": traj_batch.info["returned_episode_returns"],
            }
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


def log_losses(fixed_config, metrics):
    for i in range(1, fixed_config["NUM_UPDATES"] + 1):
        # TODO: This isn't env steps, just steps take for training
        steps = i * fixed_config["num_steps"] * fixed_config["num_envs"]
        # TODO: This is just one way to log the losses, can return to and edit later
        log_dict = {}

        # Log omce for each update (i-1)
        # Take the final epoch (which is axis 1)
        # And the mean over the minibatches for that epocj

        log_dict[f"train/total_loss"] = metrics["loss"][0][:, i - 1, -1, :].mean()
        log_dict[f"train/value_loss"] = metrics["loss"][1][0][:, i - 1, -1, :].mean()
        log_dict[f"train/loss_actor"] = metrics["loss"][1][1][:, i - 1, -1, :].mean()
        log_dict[f"train/entropy"] = metrics["loss"][1][2][:, i - 1, -1, :].mean()

        log_dict["steps"] = steps

        wandb.log(log_dict)


def log_episode_metrics(fixed_config, metrics):
    mean_completed_return_over_envs = metrics["returned_episode_returns"].mean(axis=-1)
    mean_completed_return_over_envs = mean_completed_return_over_envs.reshape(-1)
    rew_fig, ax = plt.subplots()
    plt.plot(mean_completed_return_over_envs)
    # Log as rep because any discounting etc is based on rep steps
    wandb.log({"train/mean_completed_return": rew_fig})


# Specifically used for DeMoor m=2, L=1


def plot_policy(policy_rep, policy_params):
    # For simplicity, redefine here without incluing in_transit
    # because for the simple example we can plot there is no in-transit
    # ANd having it with no shape causes problems
    @struct.dataclass
    class EnvObs:
        stock: chex.Array
        in_transit: chex.Array
        action_mask: chex.Array

        @property
        def obs(self):
            return jnp.hstack([self.stock])

    stock = jnp.array([[i, j] for i in range(0, 11) for j in range(0, 11)])
    in_transit = jnp.array([1] * 121).reshape(121, 1)[
        :, 1:
    ]  # Doing this, should get the empty array we expect
    action_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 121).reshape(121, 11)
    all_obs = EnvObs(stock=stock, action_mask=action_mask, in_transit=in_transit)

    rep_actions = jax.vmap(
        jax.vmap(policy_rep.apply_deterministic, in_axes=(0, None, None)),
        in_axes=(None, 0, None),
    )(policy_params, all_obs, jax.random.PRNGKey(1))
    rep_df = pd.DataFrame(
        {
            "age_1": all_obs.stock[:, 0],
            "age_2": all_obs.stock[:, 1],
            "action": rep_actions.reshape(-1),
        }
    )
    rep_df = rep_df.pivot(columns="age_2", index="age_1", values="action").sort_index(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    rep_heatmap = sns.heatmap(
        rep_df, annot=True, cmap="Greens_r", vmax=5, square=True, ax=ax
    )
    wandb.log({"rep/policy_plot": wandb.Image(rep_heatmap)})


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    # Resolve these to dicts because make_train adds extra elements
    # and we want to use hp_config as input to the dataclass
    fixed_config = omegaconf.OmegaConf.to_container(cfg.fixed_config, resolve=True)
    # Instantiate the issuing policy if provided
    if "issuing_policy" in fixed_config["environment"]["env_kwargs"]:
        fixed_config["environment"]["env_kwargs"][
            "issuing_policy"
        ] = hydra.utils.instantiate(
            fixed_config["environment"]["env_kwargs"]["issuing_policy"]
        )
    hp_config = HPConfig(
        **omegaconf.OmegaConf.to_container(cfg.hp_config, resolve=True)
    )
    train_vjit = jax.jit(jax.vmap(make_train(fixed_config), in_axes=(None, 0)))
    rng_train = jax.random.PRNGKey(cfg.training.seed)
    # NOTE: This is vmapping over n_seeds - i.e. multiple different training runs.
    # Be careful, can use a lot of RAM given size of stuff we have to carry along
    output = train_vjit(hp_config, jax.random.split(rng_train, cfg.training.n_seeds))

    # Loop over the output to log
    log_losses(fixed_config, output["metrics"])
    log_episode_metrics(fixed_config, output["metrics"])

    policy = hydra.utils.instantiate(cfg.policies.replenishment)
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy.apply_deterministic)
    policy_params = output["runner_state"][0].params
    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)
    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)

    log_to_wandb = {}
    log_to_wandb[f"eval/return_mean"] = fitness.mean(axis=-1)
    log_to_wandb[f"eval/return_std"] = fitness.std(axis=-1)
    for k, v in kpis.items():
        if k in cfg.environment.scalar_kpis_to_log:
            log_to_wandb[f"eval/{k}"] = v.mean(axis=-1)

    wandb.log(log_to_wandb)

    if cfg["plot_policy"]:
        plot_policy(policy, policy_params)


if __name__ == "__main__":
    main()
