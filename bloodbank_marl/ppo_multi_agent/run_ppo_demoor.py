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

    # TODO This is customised for our setup, so we should probably give it a more specific name
    # TODO: Check discounting of new_ep_return by using known policies.

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
        obs, env_state, reward, truncation, termination, _ = self._env.step(
            key, state.env_state, action, params
        )
        done = jax.lax.eq(env_state.live_agents.sum(axis=-1), 0)
        new_episode_return = state.episode_returns + (
            reward
            * (1 - env_state.agent_id)
            * (params.gamma ** jnp.clip(env_state.day - 1, 0, None))
        )  # So we only capture the reward of the first agent
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
        info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, truncation, termination, info


@struct.dataclass
class LogInfo:
    timestep: int
    returned_episode_returns: float
    returned_episode_lengths: int
    returned_episode: bool


@struct.dataclass
class Transition:
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: LogInfo


def empty_transitions(n_steps, obs_dim):
    return Transition(
        done=jnp.array([False] * n_steps, dtype=jnp.bool_),
        action=jnp.array([-1] * n_steps, dtype=jnp.int32),
        value=jnp.array([-1.0] * n_steps, dtype=jnp.float32),
        reward=jnp.array([-1.0] * n_steps, dtype=jnp.float32),
        log_prob=jnp.array([-1.0] * n_steps, dtype=jnp.float32),
        obs=jnp.array([[-1] * obs_dim] * n_steps, dtype=jnp.float32),
        info=LogInfo(
            timestep=jnp.array([-1] * n_steps, dtype=jnp.int32),
            returned_episode_returns=jnp.array([-1.0] * n_steps, dtype=jnp.float32),
            returned_episode_lengths=jnp.array([-1] * n_steps, dtype=jnp.int32),
            returned_episode=jnp.array([False] * n_steps, dtype=jnp.bool_),
        ),
    )


def update_transition_pre_step(idx, t, update):
    # Want to update obs, action, value and log prob
    _obs, _action, _value, _log_prob = update
    action = t.action.at[idx].set(_action)
    obs = t.obs.at[idx].set(_obs.obs)
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


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    # When using discrete actions, pad so that logits for each dist have same dims
    # We pad with a large negative number so that the probability of these actions
    # is effectively zero
    action_pad: int = 0
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        action_pad = jnp.hstack(
            [
                jnp.zeros(self.action_dim - self.action_pad),
                jnp.full(self.action_pad, -1e9),
            ]
        )
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = actor_mean + action_pad
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


# Maybe Flax PPO network?
class FlaxStochasticPolicy:
    def __init__(
        self,
        model_class,
        model_kwargs,
        policy_id,
        env_name=None,
        env_kwargs={},
        env_params={},
    ):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.env_params = env_params
        self.model = model_class(**model_kwargs)

    def apply(self, policy_params, obs, rng):
        # Apply should get you an action
        # TODO: We probably want to use env obs/action space when initialising the network
        pi, _ = self.model.apply(policy_params[self.policy_id], obs.obs)
        return pi.sample(seed=rng)

    def apply_for_training(self, policy_params, obs, rng):
        # Apply training should sample an action and return action, log_prob and value
        # In training, this should help us avoid issues around different action spaces
        # as long as both Discrete/both Box of same length etc.
        pi, value = self.model.apply(policy_params[self.policy_id], obs.obs)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        return action, log_prob, value

    def apply_deterministic(self, policy_params, obs, rng):
        # Get the most likely action
        pi, _ = self.model.apply(policy_params[self.policy_id], obs.obs)
        return pi.probs.argmax(axis=-1)

    def get_initial_params(self, rng):
        # TODO For now, hardcoded for DeMoor with L=1; will want to make more general
        return self.model.init(rng, jnp.zeros(self.env_kwargs["max_useful_life"]))


# We need an update epoch function for each policy. We'll pass in the network and the config for the policy
def make_update_epoch(network, config):
    def _update_epoch(update_state, unused):
        ## One whole epoch,
        def _update_minbatch(train_state, batch_info):
            ## We run this one for each minibatch
            traj_batch, advantages, targets = batch_info

            def _loss_fn(params, traj_batch, gae, targets):
                # RERUN NETWORK
                pi, value = network.model.apply(params, traj_batch.obs)
                log_prob = pi.log_prob(traj_batch.action)

                # CALCULATE VALUE LOSS
                value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                    -config["CLIP_EPS"], config["CLIP_EPS"]
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
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    * gae
                )
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean()
                entropy = pi.entropy().mean()

                total_loss = (
                    loss_actor
                    + config["VF_COEF"] * value_loss
                    - config["ENT_COEF"] * entropy
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
        batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
        assert (
            batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
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
                x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
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
    for agent in ["REP", "ISSUE"]:
        config[agent]["NUM_UPDATES"] = int(
            config[agent]["TOTAL_TIMESTEPS"]
            // config[agent]["NUM_STEPS"]
            // config[agent]["NUM_ENVS"]
        )
        config[agent]["MINIBATCH_SIZE"] = int(
            config[agent]["NUM_ENVS"]
            * config[agent]["NUM_STEPS"]
            // config[agent]["NUM_MINIBATCHES"]
        )

    # TODO There is probably a better way to enfore this but this will do for now
    assert (
        config["REP"]["NUM_ENVS"] == config["ISSUE"]["NUM_ENVS"]
    ), "Number of envs must be the same for both policies"
    assert (
        config["REP"]["NUM_UPDATES"] == config["ISSUE"]["NUM_UPDATES"]
    ), "Number of updates must be the same for both policies"

    env, env_params = make(config["ENV_NAME"])
    env = LogWrapper(env)
    # env = FlattenObservationWrapper(env)

    ## Supporting function for annealing the learning rate.
    def linear_schedule(count, config):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    # Inner, created function that we will subsequently JIT
    def train(rng):
        # INIT NETWORKS
        ## Replenishment network
        # TODO: Don't hardcode action_dim
        network_rep = FlaxStochasticPolicy(
            model_class=ActorCritic,
            model_kwargs={
                "action_dim": 11,
                "action_pad": 0,
                "activation": config["REP"]["ACTIVATION"],
            },
            policy_id=0,
            env_kwargs={"max_useful_life": 2},
        )
        rng, _rng = jax.random.split(rng)
        # init_x_rep = jnp.zeros(2)  # TODO Don't hardcode
        # network_params_rep = network_rep.model.init(_rng, init_x_rep)
        network_params_rep = network_rep.get_initial_params(_rng)
        if config["REP"]["ANNEAL_LR"]:
            tx_rep = optax.chain(
                optax.clip_by_global_norm(config["REP"]["MAX_GRAD_NORM"]),
                optax.adam(
                    learning_rate=partial(linear_schedule, config=config["REP"]),
                    eps=1e-5,
                ),
            )
        else:
            tx_rep = optax.chain(
                optax.clip_by_global_norm(config["REP"]["MAX_GRAD_NORM"]),
                optax.adam(config["REP"]["LR"], eps=1e-5),
            )
        train_state_rep = TrainState.create(
            apply_fn=network_rep.apply,
            params=network_params_rep,
            tx=tx_rep,
        )
        ## Issuing network
        # TODO: Don't hardcode action dim
        network_issue = FlaxStochasticPolicy(
            model_class=ActorCritic,
            model_kwargs={
                "action_dim": 11,
                "action_pad": 8,
                "activation": config["ISSUE"]["ACTIVATION"],
            },
            policy_id=1,
            env_kwargs={"max_useful_life": 2},
        )
        rng, _rng = jax.random.split(rng)
        network_params_issue = network_issue.get_initial_params(_rng)
        # init_x_issue = jnp.zeros(2)  # TODO: Don't hardcodde
        # network_params_issue = network_issue.model.init(_rng, init_x_issue)

        if config["ISSUE"]["ANNEAL_LR"]:
            tx_issue = optax.chain(
                optax.clip_by_global_norm(config["ISSUE"]["MAX_GRAD_NORM"]),
                optax.adam(
                    learning_rate=partial(linear_schedule, config=config["ISSUE"]),
                    eps=1e-5,
                ),
            )
        else:
            tx_issue = optax.chain(
                optax.clip_by_global_norm(config["ISSUE"]["MAX_GRAD_NORM"]),
                optax.adam(config["ISSUE"]["LR"], eps=1e-5),
            )
        train_state_issue = TrainState.create(
            apply_fn=network_issue.apply,
            params=network_params_issue,
            tx=tx_issue,
        )
        ## Policy manager with both networks
        pm = policy_manager.PolicyManager(
            [network_rep.apply_for_training, network_issue.apply_for_training]
        )

        train_state = (train_state_rep, train_state_issue)
        # INIT ENV
        ## This is pretty standard Gymnax; vmapping over the rng in reset so we can run multiple rollouts in parallele
        ## but using the same env_params
        # NOTE: At the moment this isn't really getting used
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["REP"]["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # Build update epoch functions
        _update_epoch_rep = make_update_epoch(network_rep, config["REP"])
        _update_epoch_issue = make_update_epoch(network_issue, config["ISSUE"])

        # TRAIN LOOP
        ## This outer function is what will be run NUM_UPDATES times
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            ## Single step in the environment, collecting a transition
            train_state_rep, train_state_issue, env_state, last_obs, rng = runner_state
            policy_params = {0: train_state_rep.params, 1: train_state_issue.params}

            # TODO: Right now, we're resetting the env each time we want to collect a trajectory.
            # We probably shouldn't do this; we can use env_state to start back where we stopped.
            # But this will need a bit of rewriting that is not essential for getting training to work.
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
                action, log_prob, value = pm.apply(policy_params, last_obs, _rng)
                # rng, _rng = jax.random.split(rng)
                # action = pi.sample(seed=_rng)
                # log_prob = pi.log_prob(action)

                # Update transition with last_obs and action for the agent that is about to act
                rep_idx, rep_t = jax.lax.cond(
                    last_obs.agent_id == 0,
                    update_transition_pre_step,
                    lambda idx, t, update: (idx, t),
                    rep_idx,
                    rep_t,
                    (last_obs, action, value, log_prob),
                )
                issue_idx, issue_t = jax.lax.cond(
                    last_obs.agent_id == 1,
                    update_transition_pre_step,
                    lambda idx, t, update: (idx, t),
                    issue_idx,
                    issue_t,
                    (last_obs, action, value, log_prob),
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
                rep_t = empty_transitions(n_rep, 2)
                issue_idx = 0
                issue_t = empty_transitions(n_issue, 2)
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
                    n_rep=config["REP"]["NUM_STEPS"] + 2,
                    n_issue=config["ISSUE"]["NUM_STEPS"] + 2,
                    policy_params=policy_params,
                )
            )
            vmap_collect_trajectories = jax.vmap(get_ma_samples)

            rng, _rng = jax.random.split(rng)
            _rng = jax.random.split(_rng, config["REP"]["NUM_ENVS"])
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
            last_obs_rep = trans_rep.obs[-1, :, :]
            traj_batch_rep = jax.tree_util.tree_map(lambda x: x[1:-1, ...], trans_rep)
            _, last_val_rep = network_rep.model.apply(policy_params[0], last_obs_rep)

            trans_issue = jax.tree_util.tree_map(
                lambda x: x.swapaxes(0, 1), rollout_output[3]
            )
            last_obs_issue = trans_issue.obs[-1, :, :]
            traj_batch_issue = jax.tree_util.tree_map(
                lambda x: x[1:-1, ...], trans_issue
            )

            _, last_val_issue = network_issue.model.apply(
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
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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

            _calculate_gae_rep = partial(_calculate_gae, config=config["REP"])
            _calculate_gae_issue = partial(_calculate_gae, config=config["ISSUE"])

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
                config["REP"]["UPDATE_EPOCHS"],
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
                config["ISSUE"]["UPDATE_EPOCHS"],
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
        ## Run NUM_UPDATES training updates (for now just use REP, but we have assert to make sure same for both)
        runner_state, metric = jax.lax.scan(
            _update_step,
            runner_state,
            None,
            config["REP"]["NUM_UPDATES"],
        )
        # Output from whole training process
        return {"runner_state": runner_state, "metrics": metric}

    return train


# Just temporary, quite rough, useful to be able to see policies when
# there is only a small obs space
def plot_policies(network_rep, network_issue, policy_params):
    stock = jnp.array([[i, j] for i in range(0, 11) for j in range(0, 11)])

    stock = jnp.array([[i, j] for i in range(0, 11) for j in range(0, 11)])
    in_transit = jnp.array([0] * 121).reshape(121, 1)
    agent_id = jnp.array([1] * 121).reshape(121, 1)
    all_obs = EnvObs(stock=stock, in_transit=in_transit, agent_id=agent_id)

    @struct.dataclass
    class SimpleEnvObs:
        obs: jnp.ndarray

    simple_all_obs = SimpleEnvObs(all_obs.obs[:, 1:])

    rep_actions = jax.vmap(
        jax.vmap(network_rep.apply_deterministic, in_axes=(0, None, None)),
        in_axes=(None, 0, None),
    )(policy_params, simple_all_obs, jax.random.PRNGKey(1))
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

    issue_actions = jax.vmap(
        jax.vmap(network_issue.apply_deterministic, in_axes=(0, None, None)),
        in_axes=(None, 0, None),
    )(policy_params, simple_all_obs, jax.random.PRNGKey(1))
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
        1, config["REP"]["NUM_UPDATES"] + 1
    ):  # TODO: Again using rep but we have forced them to be the same
        # These are all approximate, they ignore extra steps we have to take and
        # just count the ones we trained on. We can adjust later.
        rep_steps = i * config["REP"]["NUM_STEPS"] * config["REP"]["NUM_ENVS"]
        issue_steps = i * config["ISSUE"]["NUM_STEPS"] * config["ISSUE"]["NUM_ENVS"]
        total_steps = rep_steps + issue_steps
        # TODO: This is just one way to log the losses, can return to and edit later
        log_dict = {}

        # Log omce for each update (i-1)
        # Take the final epoch (which is axis 1)
        # And the mean over the minibatches for that epocj
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
    run = wandb.init(**config["wandb"]["init"], config=config)
    train = make_train(config)
    print("Starting training")
    output = train(jax.random.PRNGKey(config["TRAIN_SEED"]))
    log_losses(config, output["metrics"])
    log_episode_metrics(config, output["metrics"])
    print("Training complete, starting evaluation")

    fitness = GymnaxFitness(
        config["ENV_NAME"],
        5000,
        num_rollouts=10000,
        num_warmup_days=100,
        max_warmup_steps=1500,
    )
    network_rep = FlaxStochasticPolicy(
        model_class=ActorCritic,
        model_kwargs={
            "action_dim": 11,
            "action_pad": 0,
            "activation": config["REP"]["ACTIVATION"],
        },
        policy_id=0,
        env_kwargs={"max_useful_life": 2},
    )
    network_issue = FlaxStochasticPolicy(
        model_class=ActorCritic,
        model_kwargs={
            "action_dim": 11,
            "action_pad": 8,
            "activation": config["ISSUE"]["ACTIVATION"],
        },
        policy_id=1,
        env_kwargs={"max_useful_life": 2},
    )
    pm = policy_manager.PolicyManager(
        [network_rep.apply_deterministic, network_issue.apply_deterministic]
    )

    fitness.set_apply_fn(pm.apply)

    rep_params = jax.tree_util.tree_map(
        lambda x: jnp.array([x]), output["runner_state"][0].params
    )
    issue_params = jax.tree_util.tree_map(
        lambda x: jnp.array([x]), output["runner_state"][1].params
    )
    policy_params = {0: rep_params, 1: issue_params}
    eval_output = fitness.rollout(
        jax.random.PRNGKey(config["EVAL_SEED"]), policy_params
    )
    print(f"Mean return on eval episodes: {eval_output[0].mean()}")
    wandb.log({"return_mean": eval_output[0].mean()})
    wandb.log({"return_std": eval_output[0].std()})

    if config["plot_policies"]:
        plot_policies(network_rep, network_issue, policy_params)


if __name__ == "__main__":
    main()
