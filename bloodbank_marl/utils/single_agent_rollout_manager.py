# Based on our RolloutWrapper from viso_jax, in turn based on the original from gymnax

import jax
import jax.numpy as jnp
import gymnax
from functools import partial
from typing import Optional, Callable, Dict, Any, Tuple
import chex
from bloodbank_marl.utils.gymnax_fitness import make
from bloodbank_marl.scenarios.meneses_perishable.gymnax_env import (
    MenesesPerishableGymnax,
    EnvObs,
)


# TODO Sort out make statement
# TODO Consider approach that only reports cum_return (and/or some other aggregraed performance measures) so it needs less memory
# TODO Finalise how we're going to deal with observations (EnvObs vs flat arrays) - currently this uses an EnvObs to go to replenishment
# policy so we can use the versions we wrote for the multiagent env.
# TODO: At the moment we can only apply performance penalties if we collect detailed info - we might want a less memory hungry way of doing this
# e.g. by having an alternative rollout mathod than maintains a cumulative info.


# def make(env_name, **env_kwargs):
#    """Helper function to create a Gymnax environment."""
#    if env_name == "MenesesPerishableGymnax":
#        env = MenesesPerishableGymnax(**env_kwargs)
#        env_params = env.default_params
#        return env, env.default_params
#    else:
#        raise ValueError(f"Unknown environment: {env_name}")


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
        self.env, default_env_params = make(self.env_id, **env_kwargs)

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
            # NOTE: CONVENIENCE ADDED FOR NOW
            # rc_obs = EnvObs(stock=state.stock, in_transit=state.in_transit[:, 1:])
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

        output["obs"] = obs.obs[start_idx:stop_idx]
        output["action"] = action[start_idx:stop_idx]
        output["reward"] = reward[start_idx:stop_idx]
        output["next_obs"] = next_obs.obs[start_idx:stop_idx]
        output["done"] = done[start_idx:stop_idx]
        output["cum_return"] = cum_return

        if self.return_info:
            output["kpis"] = self.env.calculate_kpis(output["info"])

            # Add penalty to cum return if limits on KPIs are exceeded
            target_breached_penalty = self.env.calculate_target_kpi_penalty(
                output["kpis"], self.env_params
            )
            output["cum_return"] += target_breached_penalty

        return output

    @property
    def input_shape(self) -> Tuple[int, ...]:
        """Get the shape of the observation."""
        rng = jax.random.PRNGKey(0)
        obs, state = self.env.reset(rng, self.env_params)
        return obs.shape

    @partial(jax.jit, static_argnums=(0,))
    def single_rollout_return_only(
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
            # NOTE: CONVENIENCE ADDED FOR NOW
            # rc_obs = EnvObs(stock=state.stock, in_transit=state.in_transit[:, 1:])
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

            return carry, None

        # Scan over episode step loop
        carry_out, _ = jax.lax.scan(
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

        # Extract the discounted sum of rewards accumulated by agent in episode rollout
        cum_return = carry_out[-2]
        output["cum_return"] = cum_return
        return output

    @partial(jax.jit, static_argnums=(0,))
    def population_rollout_return_only(
        self, rng_eval: chex.PRNGKey, policy_params: chex.Array
    ) -> Dict[str, chex.Array]:
        """Reshape parameter vector and evaluate the generation."""
        # Evaluate population of nets on gymnax task - vmap over rng & params
        pop_rollout = jax.vmap(self.batch_rollout_return_only, in_axes=(None, 0))
        return pop_rollout(rng_eval, policy_params)

    @partial(jax.jit, static_argnums=(0,))
    def batch_rollout_return_only(
        self, rng_eval: chex.PRNGKey, policy_params: chex.Array
    ) -> Dict[str, chex.Array]:
        """Evaluate a generation of networks on RL/Supervised/etc. task."""
        # vmap over different MC fitness evaluations for single network
        batch_rollout = jax.vmap(self.single_rollout_return_only, in_axes=(0, None))
        return batch_rollout(rng_eval, policy_params)
