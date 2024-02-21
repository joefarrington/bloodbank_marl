# Taken from our original implementation in jupyter notebook
# multiagent_demo/20230817_jax_demoor_marl_refine.ipynb
# TODO: Put in a link to the gymnax rollout manager
# TODO Make more general, some elements of this are specific to the DeMoor environment

import gymnasium
import numpy as np
import pettingzoo
from functools import partial
import jax
import jax.numpy as jnp
import distrax

jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import DeMoorPerishableMAJAX
from bloodbank_marl.scenarios.meneses_perishable.jax_env import MenesesPerishableEnv


# For now, monkey patch the gymnax.make function
def make(env_name, **env_kwargs):
    if env_name == "MenesesPerishable":
        return (
            MenesesPerishableEnv(**env_kwargs),
            MenesesPerishableEnv().default_params,
        )
    elif env_name == "DeMoorPerishable":
        return (
            DeMoorPerishableMAJAX(**env_kwargs),
            DeMoorPerishableMAJAX().default_params,
        )
    else:
        raise ValueError(f"Unknown environment '{env_name}'")


class RolloutManager(object):
    def __init__(
        self,
        policy_manager,
        env_name,
        env_kwargs={},
        env_params={},
        num_warmup_days=0,
        max_warmup_steps=1500,
        gamma=0.99,
    ):
        self.policy_manager = policy_manager
        self.get_action = jax.vmap(self.policy_manager.apply, in_axes=(None, 0, 0))

        # Setup functionalities for vectorized batch rollout
        # TODO: Some sort of make function
        # TODO: Warm up period
        # self.env = DeMoorPerishableMAJAX(**env_kwargs)

        # Define the RL environment & replace default parameters if desired
        self.env, self.env_params = make(env_name, **env_kwargs)
        self.env_params = self.env_params.create_env_params(**env_params)

        self.num_warmup_days = num_warmup_days
        self.max_warmup_steps = max_warmup_steps

        self.gamma = gamma

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.env_params)

    @partial(jax.jit, static_argnums=0)
    def batch_end_of_warmup_reset(self, keys, states):
        return jax.vmap(self.env.end_of_warmup_reset, in_axes=(0, 0, None))(
            keys, states, self.env_params
        )

    @partial(jax.jit, static_argnums=0)
    def batch_step(self, keys, state, action):
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))(
            keys, state, action, self.env_params
        )

    @partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
    def batch_warmup_state_update(self, state, obs, next_o, next_s, warmup_done):
        """Update the state during the warmup period."""
        state = jax.tree_map(lambda x, y: jnp.where(warmup_done, x, y), state, next_s)
        obs = jax.tree_map(lambda x, y: jnp.where(warmup_done, x, y), obs, next_o)
        return state, obs

    @partial(jax.jit, static_argnums=(0, 3))
    def batch_evaluate(self, rng_input, policy_params, num_envs):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.batch_reset(jax.random.split(rng_reset, num_envs))

        @jax.vmap
        def create_empty_infos(_):
            """Dummy function to allow us to create empty infos for multiple envs"""
            return self.env.empty_infos

        @jax.vmap
        def accumulate_info(agent_id, valid_mask, step_info, cum_info):
            return jax.tree_map(
                lambda x, y: jax.lax.select(valid_mask[agent_id] == 1, x, y),
                cum_info.accumulate_infos_one_agent(agent_id, step_info),
                cum_info,
            )

        def warmup_step(state_input):
            """lax.scan compatible warm-upstep transition in jax env."""
            obs, state, rng = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)

            action = self.get_action(
                policy_params, obs, jax.random.split(rng_net, num_envs)
            )
            next_o, next_s, reward, truncation, termination, info = self.batch_step(
                jax.random.split(rng_step, num_envs),
                state,
                action,
            )
            # If we've reached the target number of days, we don't want to update the state or the obs

            warmup_done = jax.lax.ge(state.day, self.num_warmup_days)
            state, obs = self.batch_warmup_state_update(
                state, obs, next_o, next_s, warmup_done
            )
            carry = (
                obs,
                state,
                warmup_done,
                rng,
            )
            return carry

        def policy_step(state_input):
            """lax.scan compatible step transition in jax env."""
            obs, state, rng, cum_reward, cum_return, cum_info, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)

            action = self.get_action(
                policy_params, obs, jax.random.split(rng_net, num_envs)
            )
            next_o, next_s, reward, truncation, termination, info = self.batch_step(
                jax.random.split(rng_step, num_envs),
                state,
                action,
            )
            new_cum_reward = cum_reward.at[jnp.arange(num_envs), next_o.agent_id].add(
                reward * valid_mask[jnp.arange(num_envs), next_o.agent_id]
            )
            new_cum_info = accumulate_info(next_o.agent_id, valid_mask, info, cum_info)
            new_cum_return = cum_return.at[jnp.arange(num_envs)].add(
                (reward * (self.gamma ** jnp.clip((next_s.day - 1), a_min=0)))
                * (next_o.agent_id == 0)
                * valid_mask[jnp.arange(num_envs), next_o.agent_id]
            )  # Only update when replenishment agent acting
            agent_done = jax.lax.bitwise_or(truncation, termination)
            new_valid_mask = valid_mask.at[
                jnp.arange(num_envs), next_o.agent_id
            ].multiply(1 - agent_done)
            carry = (
                next_o,
                next_s,
                rng,
                new_cum_reward,
                new_cum_return,
                new_cum_info,
                new_valid_mask,
            )  # [new_valid_mask]

            return carry

        def cond_fn_warmup(carry):
            """Condition for warmup loop."""
            obs, state, warmup_done, rng = carry
            return jax.lax.bitwise_not(warmup_done)

        def cond_fn_policy_step(carry):
            """Condition for warmup loop."""
            (
                next_o,
                next_s,
                rng,
                cum_reward,
                cum_return,
                cum_info,
                valid_mask,
            ) = carry
            # When all are equal to 0, we are done
            return jax.lax.bitwise_and(
                jnp.any(jax.lax.gt(valid_mask, 0)),
                jax.lax.lt(next_s.step, self.num_env_steps),
            )

        # Run the warmup period
        carry_out = jax.lax.while_loop(
            cond_fn_warmup, warmup_step, (obs, state, False, rng_episode)
        )

        # Use the state and obs from end of warm-up period but reset other stuff in the state (reward, info etc).
        obs, state, warmup_done, rng_episode = carry_out
        rng_reset, rng_episode = jax.random.split(rng_input)
        state = self.batch_end_of_warmup_reset(
            jax.random.split(rng_reset, num_envs), state
        )

        cum_reward = jnp.zeros((num_envs, self.env.num_agents))
        cum_return = jnp.zeros(num_envs)
        cum_info = create_empty_infos(jnp.zeros(num_envs))
        valid_mask = jnp.ones((num_envs, self.env.num_agents), dtype=jnp_int)

        # Scan over episode step loop
        carry_out = jax.lax.while_loop(
            cond_fn_policy_step,
            policy_step,
            (obs, state, rng_episode, cum_reward, cum_return, cum_info, valid_mask),
        )
        cum_info = carry_out[-2]
        cum_return = carry_out[-3].squeeze()
        cum_reward = carry_out[-4].squeeze()
        return cum_reward, cum_return, cum_info
