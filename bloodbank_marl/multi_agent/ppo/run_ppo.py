# Initial basic working version of MARL for DeMoor based on PureJAXRL.
# Wuite a bit TODO, but moving this out of notebook now to version control future changes.

import jax
import jax.numpy as jnp
import distrax
from flax import struct
from functools import partial


from bloodbank_marl.scenarios.de_moor_perishable.jax_env import DeMoorPerishableMAJAX
from bloodbank_marl.policies import replenishment, issuing, policy_manager, common
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


# TODO: Automate creation of Transition based on information about the env
# Specifically, the shape of the EnvObs object (plus env_args)


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
    for agent in ["replenishment", "issuing"]:
        config["training"][agent]["num_updates"] = int(
            config["training"][agent]["total_timesteps"]
            // config["training"][agent]["num_steps"]
            // config["training"][agent]["num_envs"]
        )
        config["training"][agent]["minibatch_size"] = int(
            config["training"][agent]["num_envs"]
            * config["training"][agent]["num_steps"]
            // config["training"][agent]["num_minibatches"]
        )

    # TODO There is probably a better way to enfore this but this will do for now
    assert (
        config["training"]["replenishment"]["num_envs"]
        == config["training"]["issuing"]["num_envs"]
    ), "Number of envs must be the same for both policies"
    assert (
        config["training"]["replenishment"]["num_updates"]
        == config["training"]["issuing"]["num_updates"]
    ), "Number of updates must be the same for both policies"

    env, env_params = make(
        config["environment"]["env_name"], **config["environment"]["env_kwargs"]
    )
    default_obs, _ = env.reset(jax.random.PRNGKey(1), env_params)
    env = LogWrapper(env)
    # Use default_obs to create empty Transitions

    # env = FlattenObservationWrapper(env)

    num_actions = env.num_actions(
        0
    )  # Use agent_id for rep, as forced to be same for both agents
    action_shape = env.action_space(env_params, 0).shape

    def empty_transitions(n_steps):
        return Transition(
            done=jnp.array([False] * n_steps, dtype=jnp.bool_),
            # TODO: This also needs to be based on the env, currently for Meneses (1D for DeMoor and integer)
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
        # INIT NETWORKS
        ## Replenishment network
        policy_rep = hydra.utils.instantiate(config["policies"]["replenishment"])
        rng, _rng = jax.random.split(rng)
        network_params_rep = policy_rep.get_initial_params(_rng)
        if config["training"]["replenishment"]["anneal_lr"]:
            tx_rep = optax.chain(
                optax.clip_by_global_norm(
                    config["training"]["replenishment"]["max_grad_norm"]
                ),
                optax.adam(
                    learning_rate=partial(
                        linear_schedule, config=config["training"]["replenishment"]
                    ),
                    eps=1e-5,
                ),
            )
        else:
            tx_rep = optax.chain(
                optax.clip_by_global_norm(
                    config["training"]["replenishment"]["max_grad_norm"]
                ),
                optax.adam(config["training"]["replenishment"]["lr"], eps=1e-5),
            )
        train_state_rep = TrainState.create(
            apply_fn=policy_rep.apply,
            params=network_params_rep,
            tx=tx_rep,
        )
        ## Issuing network
        policy_issue = hydra.utils.instantiate(config["policies"]["issuing"])
        rng, _rng = jax.random.split(rng)
        network_params_issue = policy_issue.get_initial_params(_rng)

        if config["training"]["issuing"]["anneal_lr"]:
            tx_issue = optax.chain(
                optax.clip_by_global_norm(
                    config["training"]["issuing"]["max_grad_norm"]
                ),
                optax.adam(
                    learning_rate=partial(
                        linear_schedule, config=config["training"]["issuing"]
                    ),
                    eps=1e-5,
                ),
            )
        else:
            tx_issue = optax.chain(
                optax.clip_by_global_norm(
                    config["training"]["issuing"]["max_grad_norm"]
                ),
                optax.adam(config["training"]["issuing"]["lr"], eps=1e-5),
            )
        train_state_issue = TrainState.create(
            apply_fn=policy_issue.apply,
            params=network_params_issue,
            tx=tx_issue,
        )
        ## Policy manager with both networks
        pm = policy_manager.PolicyManager(
            [policy_rep.apply_for_training, policy_issue.apply_for_training]
        )

        train_state = (train_state_rep, train_state_issue)
        # INIT ENV
        ## This is pretty standard Gymnax; vmapping over the rng in reset so we can run multiple rollouts in parallele
        ## but using the same env_params
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(
            _rng, config["training"]["replenishment"]["num_envs"]
        )
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Build update epoch functions
        _update_epoch_rep = make_update_epoch(
            policy_rep, config["training"]["replenishment"]
        )
        _update_epoch_issue = make_update_epoch(
            policy_issue, config["training"]["issuing"]
        )

        # TRAIN LOOP
        ## This outer function is what will be run num_updates times
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            ## Single step in the environment, collecting a transition
            train_state_rep, train_state_issue, env_state, last_obs, rng = runner_state
            policy_params = {0: train_state_rep.params, 1: train_state_issue.params}

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
            get_ma_samples = jax.jit(
                partial(
                    _collect_ma_trajectories,
                    n_rep=config["training"]["replenishment"]["num_steps"] + 2,
                    n_issue=config["training"]["issuing"]["num_steps"] + 2,
                    policy_params=policy_params,
                )
            )
            vmap_collect_trajectories = jax.vmap(get_ma_samples)

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(
                _rng, config["training"]["replenishment"]["num_envs"]
            )
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
            _, last_val_rep = policy_rep.model.apply(policy_params[0], last_obs_rep)

            trans_issue = jax.tree_util.tree_map(
                lambda x: x.swapaxes(0, 1), rollout_output[3]
            )
            last_obs_issue = jax.tree_util.tree_map(lambda x: x[-1], trans_issue.obs)
            traj_batch_issue = jax.tree_util.tree_map(
                lambda x: x[1:-1, ...], trans_issue
            )

            _, last_val_issue = policy_issue.model.apply(
                policy_params[1],
                last_obs_issue,
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

            _calculate_gae_rep = partial(
                _calculate_gae, config=config["training"]["replenishment"]
            )
            _calculate_gae_issue = partial(
                _calculate_gae, config=config["training"]["issuing"]
            )

            ## Using the trajectories and the value of the last observation, calculate the advantages and targets
            advantages_rep, targets_rep = _calculate_gae_rep(
                traj_batch_rep, last_val_rep
            )
            advantages_issue, targets_issue = _calculate_gae_issue(
                traj_batch_issue, last_val_issue
            )

            # Run epochs for replenishment
            rng, _rng = jax.random.split(rng)
            update_state_rep = (
                train_state_rep,
                traj_batch_rep,
                advantages_rep,
                targets_rep,
                _rng,
            )
            update_state_rep, loss_info_rep = jax.lax.scan(
                _update_epoch_rep,
                update_state_rep,
                None,
                config["training"]["replenishment"]["update_epochs"],
            )
            train_state_rep = update_state_rep[0]
            # TODO: Not collecting info at the moment
            metric_rep = traj_batch_rep.info
            # TODO: Sort this out, will be needed when we want to continue collection instead of resetting

            # And for issuing
            rng, _rng = jax.random.split(rng)
            update_state_issue = (
                train_state_issue,
                traj_batch_issue,
                advantages_issue,
                targets_issue,
                _rng,
            )
            update_state_issue, loss_info_issue = jax.lax.scan(
                _update_epoch_issue,
                update_state_issue,
                None,
                config["training"]["issuing"]["update_epochs"],
            )
            train_state_issue = update_state_issue[0]
            metric_issue = None  # traj_batch_issue.info For now, we're just taking info for replenishment so we don't duplicate

            runner_state = (
                train_state_rep,
                train_state_issue,
                env_state,
                last_obs,
                rng,
            )
            metric = {
                "rep": {
                    "loss": loss_info_rep,
                    # TODO might want to grab some more metrics here
                    "returned_episode_returns": metric_rep.returned_episode_returns,
                },
                "issue": {"loss": loss_info_issue},
            }

            return runner_state, metric

        rng, _rng = jax.random.split(rng)

        ## Create an initial runner_state from network set-up and env reset
        runner_state = (train_state_rep, train_state_issue, env_state, obsv, _rng)
        ## Run num_updates training updates (for now just use REP, but we have assert to make sure same for both)
        runner_state, metric = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            config["training"]["replenishment"]["num_updates"],
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
        stock=stock, agent_id=agent_id, action_mask=action_mask, in_transit=in_transit
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
    for i in range(
        1, config["training"]["replenishment"]["num_updates"] + 1
    ):  # TODO: Again using rep but we have forced them to be the same
        # These are all approximate, they ignore extra steps we have to take and
        # just count the ones we trained on. We can adjust later.
        rep_steps = (
            i
            * config["training"]["replenishment"]["num_steps"]
            * config["training"]["replenishment"]["num_envs"]
        )
        issue_steps = (
            i
            * config["training"]["issuing"]["num_steps"]
            * config["training"]["issuing"]["num_envs"]
        )
        total_steps = rep_steps + issue_steps
        # TODO: Simple logging of losses, may be worth thinking about it more detail
        log_dict = {}

        # Log omce for each update (i-1)
        # Take the final epoch (which is axis 1)
        # And the mean over the minibatches for that epoch
        for a in ["rep", "issue"]:
            log_dict[f"{a}/total_loss"] = metrics[a]["loss"][0][i - 1, -1, :].mean()
            log_dict[f"{a}/value_loss"] = metrics[a]["loss"][1][0][i - 1, -1, :].mean()
            log_dict[f"{a}/loss_actor"] = metrics[a]["loss"][1][1][i - 1, -1, :].mean()
            log_dict[f"{a}/entropy"] = metrics[a]["loss"][1][2][i - 1, -1, :].mean()

        log_dict["rep/steps"] = rep_steps
        log_dict["issue/steps"] = issue_steps
        log_dict["total_steps"] = total_steps

        wandb.log(log_dict)


def log_episode_metrics(config, metrics):
    mean_completed_return_over_envs = metrics["rep"]["returned_episode_returns"].mean(
        axis=-1
    )
    mean_completed_return_over_envs = mean_completed_return_over_envs.reshape(-1)
    rew_fig, ax = plt.subplots()
    plt.plot(mean_completed_return_over_envs)
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

    fitness = hydra.utils.instantiate(cfg.evaluation.test_evaluator)

    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    pm = policy_manager.PolicyManager(
        [policy_rep.apply_deterministic, policy_issue.apply_deterministic]
    )

    fitness.set_apply_fn(pm.apply)

    rep_params = jax.tree_util.tree_map(
        lambda x: jnp.array([x]), output["runner_state"][0].params
    )
    issue_params = jax.tree_util.tree_map(
        lambda x: jnp.array([x]), output["runner_state"][1].params
    )
    policy_params = {0: rep_params, 1: issue_params}
    fitness, cum_infos, kpis = fitness.rollout(
        jax.random.PRNGKey(cfg.evaluation.seed), policy_params
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
