# Initial basic working version of MARL for DeMoor based on PureJAXRL.
# Wuite a bit TODO, but moving this out of notebook now to version control future changes.

# NOTE
# -> We're still collecting trajectories for both, even through only one trained, because easiest to report
#    metrics based on replenishment steps
# -> This is hardcoded for two agents, but could be extended to more
# -> We'll need to create a more complex wrapper for heuristic policies so they can be used within the training collected process;
#    this will probably be the most general case so may just use it everywhere for simplicity

import jax
import jax.numpy as jnp
import distrax
from flax import struct
from functools import partial


from bloodbank_marl.scenarios.de_moor_perishable.jax_env import DeMoorPerishableMAJAX
from bloodbank_marl.policies import policy_manager
from bloodbank_marl.utils.gymnax_fitness import GymnaxFitness
from bloodbank_marl.utils.rollout_manager import RolloutManager
from bloodbank_marl.utils.gymnax_wrappers import LogEnvState, LogWrapper, LogInfo

# from bloodbank_marl.ppo_multi_agent.ppo_models import DiscreteActorCritic
# from bloodbank_marl.ppo_multi_agent.ppo_policies import FlaxStochasticPolicy
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState

import gymnax
import chex
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces
from bloodbank_marl.utils.gymnax_fitness import make
import hydra
import omegaconf
import wandb
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import EnvObs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly


@struct.dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: EnvObs
    info: LogInfo


def update_transition_pre_step(idx, t, update):
    # Want to update obs, action, value and log prob
    _obs, _action, _value, _log_prob = update
    action = t.action.at[idx].set(_action)
    obs = t.obs.replace(
        agent_id=t.obs.agent_id.at[idx].set(_obs.agent_id),
        in_transit=t.obs.in_transit.at[idx].set(_obs.in_transit),
        stock=t.obs.stock.at[idx].set(_obs.stock),
        action_mask=t.obs.action_mask.at[idx].set(_obs.action_mask),
    )
    value = t.value.at[idx].set(_value)
    log_prob = t.log_prob.at[idx].set(_log_prob)
    return idx, t.replace(obs=obs, action=action, value=value, log_prob=log_prob)


def update_transition_post_step_with_info(idx, t, update):
    _reward, _done, _info = update
    # done
    done = t.done.at[idx].set(_done)
    # reward
    reward = t.reward.at[idx].set(_reward)
    # info
    info = t.info.replace(
        timestep=t.info.timestep.at[idx].set(_info["timestep"]),
        returned_episode_returns=t.info.returned_episode_returns.at[idx].set(
            _info["returned_episode_returns"]
        ),
        returned_episode_lengths=t.info.returned_episode_lengths.at[idx].set(
            _info["returned_episode_lengths"]
        ),
        returned_episode=t.info.returned_episode.at[idx].set(_info["returned_episode"]),
    )
    idx += 1
    return idx, t.replace(done=done, reward=reward, info=info)


def update_transition_post_step(idx, t, update):
    _reward, _done = update
    # done
    done = t.done.at[idx].set(_done)
    # reward
    reward = t.reward.at[idx].set(_reward)
    # info
    # We don't log info for issuing; would just be duplication
    idx += 1
    return idx, t.replace(done=done, reward=reward)


# We need an update epoch function for each policy. We'll pass in the network and the config for the policy
def make_update_epoch(policy, config):
    def _update_epoch(update_state, unused):
        ## One whole epoch,
        def _update_minbatch(train_state, batch_info):
            ## We run this one for each minibatch
            traj_batch, advantages, targets = batch_info

            def _loss_fn(params, traj_batch, gae, targets):
                # RERUN NETWORK
                params = {policy.policy_id: params}
                log_prob, value, entropy = policy.apply_for_loss_fn(
                    params, traj_batch.obs, traj_batch.action
                )
                # log_prob = pi.log_prob(traj_batch.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                    -config["clip_eps"], config["clip_eps"]
                )
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
                        1.0 - config["clip_eps"],
                        1.0 + config["clip_eps"],
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()

                total_loss = (
                    loss_actor
                    + config["vf_coef"] * value_loss
                    - config["ent_coef"] * entropy
                )
                return total_loss, (value_loss, loss_actor, entropy)

            grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        ## Unpack update state passed into _update_epoch
        train_state, traj_batch, advantages, targets, rng = update_state
        rng, _rng = jax.random.split(rng)
        batch_size = config["minibatch_size"] * config["num_minibatches"]
        assert (
            batch_size == config["num_steps"] * config["num_envs"]
        ), "batch size must be equal to number of steps * number of envs"
        ## Create shuffled minibatches from the collected trjectories
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
                [config["num_minibatches"], config["minibatch_size"]]
                + list(x.shape[1:]),
            ),
            shuffled_batch,
        )

        ## Train on each minibatch
        train_state, total_loss = jax.lax.scan(
            _update_minbatch, train_state, minibatches
        )
        ## Pack up the results of the epoch, some will be used for next training epoch (e.g. train state)
        update_state = (train_state, traj_batch, advantages, targets, rng)
        return update_state, total_loss

    return _update_epoch


def make_train(config):
    ## Make a training function
    ## Setting up the environment and the the paramters that are going to control the size of the arrays of collected
    ## experience and the size of batches for training. These all need to be calculated and fixed so that the resulting
    ## functions can be jitted.
    ## Hyperparams set here are not things that we could vmap over because they relate to arrays sizes/
    config["training"]["num_updates"] = int(
        config["training"]["total_timesteps"]
        // config["training"]["num_steps"]
        // config["training"]["num_envs"]
    )
    config["training"]["minibatch_size"] = int(
        config["training"]["num_envs"]
        * config["training"]["num_steps"]
        // config["training"]["num_minibatches"]
    )

    env, env_params = make(
        config["environment"]["env_name"], **config["environment"]["env_kwargs"]
    )
    default_obs, _ = env.reset(jax.random.PRNGKey(1), env_params)
    env = LogWrapper(env)
    # env = FlattenObservationWrapper(env)

    num_actions = env.num_actions(
        0
    )  # Use agent_id for rep, as forced to be same for both agents
    action_shape = env.action_space(env_params, 0).shape

    def empty_transitions(n_steps):
        return Transition(
            done=jnp.array([False] * n_steps, dtype=jnp.bool_),
            action=-1
            * jnp.ones(
                (n_steps,) + action_shape
            ),  # Use agent_id=0 because both agents have the same action space
            value=jnp.array([-1.0] * n_steps, dtype=jnp.float32),
            reward=jnp.array([-1.0] * n_steps, dtype=jnp.float32),
            log_prob=jnp.array([-1.0] * n_steps, dtype=jnp.float32),
            obs=default_obs.create_empty_obs(
                config["environment"]["env_kwargs"], num_actions, n_steps
            ),
            info=LogInfo(
                timestep=jnp.array([-1] * n_steps, dtype=jnp.int32),
                returned_episode_returns=jnp.array([-1.0] * n_steps, dtype=jnp.float32),
                returned_episode_lengths=jnp.array([-1] * n_steps, dtype=jnp.int32),
                returned_episode=jnp.array([False] * n_steps, dtype=jnp.bool_),
            ),
        )

    ## Supporting function for annealing the learning rate.
    def linear_schedule(count, config):
        frac = (
            1.0
            - (count // (config["num_minibatches"] * config["update_epochs"]))
            / config["num_updates"]
        )
        return config["lr"] * frac

    # Inner, created function that we will subsequently JIT
    def train(rng):
        # INIT NETWORK
        policy_rep = hydra.utils.instantiate(config["replenishment"]["policy"])
        policy_issue = hydra.utils.instantiate(config["issuing"]["policy"])

        policy_to_train = [policy_rep, policy_issue][
            config["training"]["policy_to_train"]
        ]  # 0 or 1

        heuristic_params = jnp.array(
            config["training"]["heuristic_params"]
        )  # TODO: probably a better way to do this

        rng, _rng = jax.random.split(rng)
        network_params = policy_to_train.get_initial_params(_rng)
        if config["training"]["anneal_lr"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["training"]["max_grad_norm"]),
                optax.adam(
                    learning_rate=partial(linear_schedule, config=config["training"]),
                    eps=1e-5,
                ),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["training"]["max_grad_norm"]),
                optax.adam(config["training"]["lr"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=policy_to_train.apply,
            params=network_params,
            tx=tx,
        )
        pm = policy_manager.PolicyManager(
            [policy_rep.apply_for_training, policy_issue.apply_for_training]
        )

        # INIT ENV
        ## This is pretty standard Gymnax; vmapping over the rng in reset so we can run multiple rollouts in parallele
        ## but using the same env_params
        # NOTE: At the moment this isn't really getting used
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["training"]["num_envs"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Build update epoch functions
        _update_epoch = make_update_epoch(policy_to_train, config["training"])

        # TRAIN LOOP
        ## This outer function is what will be run num_updates times
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            ## Single step in the environment, collecting a transition
            train_state, env_state, last_obs, rng = runner_state
            policy_params = {
                config["training"]["policy_to_train"]: train_state.params,
                1
                - config["training"][
                    "policy_to_train"
                ]: heuristic_params,  # Only two agents, so this is either 0 or 1
            }

            def _env_step(vals):
                (
                    rep_idx,
                    rep_t,
                    issue_idx,
                    issue_t,
                    step,
                    last_obs,
                    last_env_state,
                    policy_params,
                    rng,
                ) = vals
                rng, _rng = jax.random.split(rng)
                action, tr_action, log_prob, value = pm.apply(
                    policy_params, last_obs, _rng
                )

                # Update transition with last_obs and action for the agent that is about to act
                rep_idx, rep_t = jax.lax.cond(
                    last_obs.agent_id == 0,
                    update_transition_pre_step,
                    lambda idx, t, update: (idx, t),
                    rep_idx,
                    rep_t,
                    (last_obs, tr_action, value, log_prob),
                )
                issue_idx, issue_t = jax.lax.cond(
                    last_obs.agent_id == 1,
                    update_transition_pre_step,
                    lambda idx, t, update: (idx, t),
                    issue_idx,
                    issue_t,
                    (last_obs, tr_action, value, log_prob),
                )

                # Take a step in the environment
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, truncation, terminination, info = env.step(
                    _rng, last_env_state, action, env_params
                )
                done = jax.lax.bitwise_or(truncation, terminination)

                # Update transition with reward, done, and info for the last step of the agent that will act next
                rep_idx, rep_t = jax.lax.cond(
                    obsv.agent_id == 0,
                    update_transition_post_step_with_info,
                    lambda idx, t, update: (idx, t),
                    rep_idx,
                    rep_t,
                    (reward, done, info),
                )
                issue_idx, issue_t = jax.lax.cond(
                    obsv.agent_id == 1,
                    update_transition_post_step,
                    lambda idx, t, update: (idx, t),
                    issue_idx,
                    issue_t,
                    (reward, done),
                )

                # If we try to set at an index longer than the array then there should be no change
                # For safety, we could also check that the index is less than the length of the array as well as the correct agent

                step += 1
                return (
                    rep_idx,
                    rep_t,
                    issue_idx,
                    issue_t,
                    step,
                    obsv,
                    env_state,
                    policy_params,
                    rng,
                )

            # This tells us when to stop collecting, when we have enough steps for both types of agent
            def cond_fn_base(vals, n_rep, n_issue):
                (
                    rep_idx,
                    rep_t,
                    issue_idx,
                    issue_t,
                    step,
                    last_obs,
                    last_env_state,
                    policy_params,
                    rng,
                ) = vals
                return jax.lax.bitwise_or(
                    jax.lax.lt(rep_idx, n_rep), jax.lax.lt(issue_idx, n_issue)
                )

            # This does the collection for a single env, using the while loop to collect until we have enough steps

            def _collect_ma_trajectories(
                rng, last_obs, last_env_state, n_rep, n_issue, policy_params
            ):
                rep_idx = 0
                rep_t = empty_transitions(n_rep)
                issue_idx = 0
                issue_t = empty_transitions(n_issue)
                step = 0
                vals = (
                    rep_idx,
                    rep_t,
                    issue_idx,
                    issue_t,
                    step,
                    last_obs,
                    last_env_state,
                    policy_params,
                    rng,
                )
                cond_fn = partial(cond_fn_base, n_rep=n_rep, n_issue=n_issue)
                return jax.lax.while_loop(cond_fn, _env_step, vals)

            # TODO: Make sure we aren't spending time rejitting etc each training iteration
            # NOTE: Add two steps, as below we remove the first and last.
            # TODO: Would be better to just be able to customize number of steps for the one being optimized, rather than both.
            get_ma_samples = jax.jit(
                partial(
                    _collect_ma_trajectories,
                    n_rep=config["replenishment"]["num_steps"] + 2,
                    n_issue=config["issuing"]["num_steps"] + 2,
                    policy_params=policy_params,
                )
            )
            vmap_collect_trajectories = jax.vmap(get_ma_samples)

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, config["training"]["num_envs"])
            rollout_output = vmap_collect_trajectories(_rng, last_obs, env_state)
            # Last obs here is the last observation, will be one agent or the other
            # So, we take the last observation in each trajectory as "last obs" for the purposes of calculating last value and GAE
            last_obs, env_state = rollout_output[5], rollout_output[6]

            # TODO: We might want to store the other elements of rollout_outputs, they will tell us the total number of steps we did
            # To match the exisiting work, we want to change this so that the first axis is number of steps,
            # then no of envs, then size of thing
            # For last obs, take the final observation
            # Also, remove the first transition
            trans_rep = jax.tree_util.tree_map(
                lambda x: x.swapaxes(0, 1), rollout_output[1]
            )
            last_obs_rep = jax.tree_util.tree_map(lambda x: x[-1], trans_rep.obs)
            traj_batch_rep = jax.tree_util.tree_map(lambda x: x[1:-1, ...], trans_rep)

            trans_issue = jax.tree_util.tree_map(
                lambda x: x.swapaxes(0, 1), rollout_output[3]
            )
            last_obs_issue = jax.tree_util.tree_map(lambda x: x[-1], trans_issue.obs)
            traj_batch_issue = jax.tree_util.tree_map(
                lambda x: x[1:-1, ...], trans_issue
            )

            traj_batches = {0: traj_batch_rep, 1: traj_batch_issue}
            last_observations = {0: last_obs_rep, 1: last_obs_issue}

            traj_batch_training = traj_batches[config["training"]["policy_to_train"]]
            last_obs_training = last_observations[config["training"]["policy_to_train"]]

            _, last_val_training = policy_to_train.model.apply(
                policy_params[config["training"]["policy_to_train"]], last_obs_training
            )

            def _calculate_gae(traj_batch, last_val, config):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["gamma"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["gamma"] * config["gae_lambda"] * (1 - done) * gae
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

            # Using the trajectories and the value of the last observation, calculate the advantages and targets
            advantages, targets = _calculate_gae(
                traj_batch_training, last_val_training, config["training"]
            )

            # Run epochs for replenishment
            rng, _rng = jax.random.split(rng)
            update_state = (
                train_state,
                traj_batch_training,
                advantages,
                targets,
                _rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                config["training"]["update_epochs"],
            )
            train_state = update_state[0]
            # TODO: Not collecting info at the moment
            metric_rep = traj_batch_rep.info
            metric_issue = None

            runner_state = (
                train_state,
                env_state,
                last_obs,
                rng,
            )
            metric = {
                "loss": loss_info,
                "returned_episode_returns": metric_rep.returned_episode_returns,
            }

            return runner_state, metric

        rng, _rng = jax.random.split(rng)

        ## Create an initial runner_state from network set-up and env reset
        runner_state = (train_state, env_state, obsv, _rng)
        ## Run num_updates training updates (for now just use REP, but we have assert to make sure same for both)
        runner_state, metric = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            config["training"]["num_updates"],
        )
        # Output from whole training process
        return {"runner_state": runner_state, "metrics": metric}

    return train


# Just temporary, quite rough, useful to be able to see policies when
# there is only a small obs space
def plot_policies(policy_rep, policy_issue, policy_params):
    # For simplicity, redefine here without incluing in_transit
    # because for the simple example we can plot there is no in-transit
    # ANd having it with no shape causes problems
    @struct.dataclass
    class EnvObs:
        agent_id: int  # Following Tianhou;
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
    agent_id = jnp.array([1] * 121).reshape(121, 1)
    action_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 121).reshape(121, 11)
    all_obs = EnvObs(
        stock=stock,
        agent_id=agent_id,
        action_mask=action_mask,
        in_transit=in_transit,
    )

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

    # Different action mask for issuing
    action_mask = jnp.hstack(
        [jnp.ones((121, 1)), jnp.where(stock > 0, 1, 0), jnp.zeros((121, 8))]
    )
    all_obs = EnvObs(
        stock=stock, agent_id=agent_id, action_mask=action_mask, in_transit=in_transit
    )

    issue_actions = jax.vmap(
        jax.vmap(policy_issue.apply_deterministic, in_axes=(0, None, None)),
        in_axes=(None, 0, None),
    )(policy_params, all_obs, jax.random.PRNGKey(1))
    issue_df = pd.DataFrame(
        {
            "age_1": all_obs.stock[:, 0],
            "age_2": all_obs.stock[:, 1],
            "action": issue_actions.reshape(-1),
        }
    )
    issue_df = issue_df.pivot(
        columns="age_2", index="age_1", values="action"
    ).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    issue_heatmap = sns.heatmap(
        issue_df, annot=True, cmap="Greens_r", vmax=5, square=True, ax=ax
    )
    wandb.log({"issue/policy_plot": wandb.Image(issue_heatmap)})


def log_losses(config, metrics):
    for i in range(1, config["training"]["num_updates"] + 1):
        # TODO: This isn't env steps, just steps take for training
        steps = i * config["training"]["num_steps"] * config["training"]["num_envs"]
        # TODO: This is just one way to log the losses, can return to and edit later
        log_dict = {}

        # Log omce for each update (i-1)
        # Take the final epoch (which is axis 1)
        # And the mean over the minibatches for that epocj

        log_dict[f"total_loss"] = metrics["loss"][0][i - 1, -1, :].mean()
        log_dict[f"value_loss"] = metrics["loss"][1][0][i - 1, -1, :].mean()
        log_dict[f"loss_actor"] = metrics["loss"][1][1][i - 1, -1, :].mean()
        log_dict[f"entropy"] = metrics["loss"][1][2][i - 1, -1, :].mean()

        log_dict["steps"] = steps

        wandb.log(log_dict)


def log_episode_metrics(config, metrics):
    mean_completed_return_over_envs = metrics["returned_episode_returns"].mean(axis=-1)
    mean_completed_return_over_envs = mean_completed_return_over_envs.reshape(-1)
    rew_fig, ax = plt.subplots()
    plt.plot(mean_completed_return_over_envs)
    # Log as rep because any discounting etc is based on rep steps
    wandb.log({"rep/mean_completed_return": rew_fig})


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True)
    run = wandb.init(**cfg.wandb.init, config=config)
    train = make_train(config)
    print("Starting training")
    output = train(jax.random.PRNGKey(cfg.training.seed))
    log_losses(config, output["metrics"])
    log_episode_metrics(config, output["metrics"])
    print("Training complete, starting evaluation")

    fitness = hydra.utils.instantiate(cfg.test_evaluator)

    policy_rep = hydra.utils.instantiate(cfg.replenishment.policy)
    policy_issue = hydra.utils.instantiate(cfg.issuing.policy)
    pm = policy_manager.PolicyManager(
        [policy_rep.apply_deterministic, policy_issue.apply_deterministic]
    )

    fitness.set_apply_fn(pm.apply)

    training_params = jax.tree_util.tree_map(
        lambda x: jnp.array([x]), output["runner_state"][0].params
    )
    # heuristic_params = jnp.array([cfg.training["heuristic_params"]])

    policy_params = {
        cfg.training["policy_to_train"]: training_params,
        # 1 - cfg.training["policy_to_train"]: heuristic_params,
    }

    fitness, cum_infos, kpis = fitness.rollout(
        jax.random.PRNGKey(cfg.evaluation_seed), policy_params
    )
    print(f"Mean return on eval episodes: {fitness.mean()}")
    wandb.log({"eval/return_mean": fitness[0].mean()})
    wandb.log({"eval/return_std": fitness[0].std()})
    test_kpis = {
        f"eval/{k}": v.mean(axis=-1)
        for k, v in kpis.items()
        if k in cfg.environment.kpis_log_eval
    }
    wandb.log(test_kpis)

    if config["plot_policies"]:
        plot_policies(policy_rep, policy_issue, policy_params)


if __name__ == "__main__":
    main()
