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


class RolloutManager(object):
    def __init__(
        self,
        policy_manager,
        env_kwargs={},
        env_params={},
        num_warmup_days=0,
        max_warmup_steps=1500,
        gamma=0.99,
    ):
        self.policy_manager = policy_manager

        # Setup functionalities for vectorized batch rollout
        # TODO: Some sort of make function
        # TODO: Warm up period
        self.env = DeMoorPerishableMAJAX(**env_kwargs)

        self.num_warmup_days = num_warmup_days
        self.max_warmup_steps = max_warmup_steps

        self.gamma = gamma

        default_env_params = self.env.default_params
        self.env_params = default_env_params.replace(**env_params)

    @partial(jax.jit, static_argnums=0)
    def batch_reset(self, keys):
        return jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.env_params)

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

        def warmup_step(state_input, _):
            """lax.scan compatible warm-upstep transition in jax env."""
            obs, state, rng = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)

            action = self.policy_manager(obs, policy_params)
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
            carry, y = [
                obs,
                state,
                rng,
            ], [next_s, state]
            return carry, y

        def policy_step(state_input, _):
            """lax.scan compatible step transition in jax env."""
            obs, state, rng, cum_reward, cum_return, cum_info, valid_mask = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)

            action = self.policy_manager(obs, policy_params)
            next_o, next_s, reward, truncation, termination, info = self.batch_step(
                jax.random.split(rng_step, num_envs),
                state,
                action,
            )
            new_cum_reward = cum_reward.at[jnp.arange(num_envs), next_o.agent_id].add(
                reward * valid_mask[jnp.arange(num_envs), next_o.agent_id]
            )
            new_cum_info = cum_info.at[jnp.arange(num_envs), next_o.agent_id, :].add(
                next_s.infos[jnp.arange(num_envs), next_o.agent_id, :]
                * jnp.expand_dims(valid_mask[jnp.arange(num_envs), next_o.agent_id], -1)
            )
            new_cum_return = cum_return.at[jnp.arange(num_envs)].add(
                (reward * (self.gamma ** jnp.clip((next_s.day - 1), a_min=0)))
                * (next_o.agent_id == 0)
                * valid_mask[jnp.arange(num_envs), next_o.agent_id]
            )  # Only update when replenishment agent acting
            agent_done = jax.lax.bitwise_or(truncation, termination)
            new_valid_mask = valid_mask.at[
                jnp.arange(num_envs), next_o.agent_id
            ].multiply(1 - agent_done)
            carry, y = [
                next_o,
                next_s,
                rng,
                new_cum_reward,
                new_cum_return,
                new_cum_info,
                new_valid_mask,
            ], [
                valid_mask,
                jnp.zeros((num_envs, 2))
                .at[jnp.arange(num_envs), next_o.agent_id]
                .add(reward),
            ]  # [new_valid_mask]

            return carry, y

        # Run the warmup period
        carry_out, scan_out_wu = jax.lax.scan(
            warmup_step,
            [
                obs,
                state,
                rng_episode,
            ],
            (),
            self.max_warmup_steps,
        )

        # Use the state and obs from end of warm-up period but reset other stuff in the state (reward, info etc).
        # TODO Create a method in the env so that this does not need to include elements from env (e.g. in stock/in-transit)
        obs, state, rng_episode = carry_out
        rng_reset, rng_episode = jax.random.split(rng_input)
        _, state_reset = self.batch_reset(jax.random.split(rng_reset, num_envs))
        state = state_reset.replace(stock=state.stock, in_transit=state.in_transit)

        # Scan over episode step loop
        carry_out, scan_out = jax.lax.scan(
            policy_step,
            [
                obs,
                state,
                rng_episode,
                jnp.zeros(
                    (num_envs, 2)
                ),  # * [[0.0] * 2]), # TODO Make dynamic based on number of agents
                jnp.zeros(num_envs),
                jnp.zeros(
                    (num_envs, 2, 5)
                ),  # TODO Make dynamic based on number of agents, size of info
                jnp.ones(
                    (num_envs, 2), dtype=jnp_int
                ),  # * [[1] * 2]), # TODO Make dynamic based on number of agents
            ],
            (),
            self.env_params.max_steps_in_episode,
        )
        cum_info = carry_out[-2].squeeze()
        cum_return = carry_out[-3].squeeze()
        cum_reward = carry_out[-4].squeeze()
        return cum_reward, cum_return, cum_info
