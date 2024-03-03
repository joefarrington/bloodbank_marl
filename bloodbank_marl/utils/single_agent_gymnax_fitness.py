# Taken from our original implementation in jupyter notebook
# multiagent_demo/20230821_jax_demorr_marl_evosax_fitness.ipynb
# TODO: Put in a link to the general evosax gymnax fitness class

import jax
import chex
from typing import Tuple, Union, Optional
from functools import partial
from flax import struct
import jax.numpy as jnp
from gymnax.environments import spaces
import numpy as np
import evosax
import gymnax
from flax import linen as nn
from evosax import OpenES, PGPE, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
from evosax.problems import GymnaxFitness
import distrax
import functools
from bloodbank_marl.scenarios.de_moor_perishable.gymnax_env import (
    DeMoorPerishableGymnax,
)
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import DeMoorPerishableMAJAX
from bloodbank_marl.scenarios.meneses_perishable.jax_env import MenesesPerishableEnv
from bloodbank_marl.scenarios.meneses_perishable.gymnax_env import (
    MenesesPerishableGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.gymnax_env import RSPerishableGymnax
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_four import (
    RSPerishableFourGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_two import (
    RSPerishableTwoGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_one import (
    RSPerishableOneGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.jax_env import RSPerishableEnv
from bloodbank_marl.scenarios.rs_perishable.jax_env_two import RSPerishableTwoEnv
from bloodbank_marl.scenarios.mirjalili_perishable_platelet.gymnax_env import (
    MirjaliliPerishablePlateletGymnax,
)
from bloodbank_marl.scenarios.simple_two_product.gymnax_env import (
    SimpleTwoProductPerishableGymnax,
)
from bloodbank_marl.scenarios.simple_two_product.jax_env import (
    SimpleTwoProductPerishableEnv,
)
from bloodbank_marl.scenarios.simple_two_product.jax_env_limit_demand import (
    SimpleTwoProductPerishableLimitDemandEnv,
)
from bloodbank_marl.scenarios.simple_two_product.gymnax_env_limit_demand import (
    SimpleTwoProductPerishableLimitDemandGymnax,
)
from bloodbank_marl.scenarios.simple_two_product.gymnax_env_try_issue_too import (
    SimpleTwoProductPerishableIncIssueGymnax,
)

jnp_int = jnp.int64 if jax.config.jax_enable_x64 else jnp.int32


# For now, monkey patch the gymnax.make function
def make(env_name, **env_kwargs):
    if env_name == "MenesesPerishable":
        return (
            MenesesPerishableEnv(**env_kwargs),
            MenesesPerishableEnv().default_params,
        )
    elif env_name == "MenesesPerishableGymnax":
        return (
            MenesesPerishableGymnax(**env_kwargs),
            MenesesPerishableGymnax().default_params,
        )
    elif env_name == "RSPerishable":
        return (
            RSPerishableEnv(**env_kwargs),
            RSPerishableEnv().default_params,
        )

    elif env_name == "RSPerishableTwo":
        return (
            RSPerishableTwoEnv(**env_kwargs),
            RSPerishableTwoEnv().default_params,
        )

    elif env_name == "RSPerishableGymnax":
        return (
            RSPerishableGymnax(**env_kwargs),
            RSPerishableGymnax().default_params,
        )

    elif env_name == "RSPerishableFourGymnax":
        return (
            RSPerishableFourGymnax(**env_kwargs),
            RSPerishableFourGymnax().default_params,
        )
    elif env_name == "RSPerishableTwoGymnax":
        return (
            RSPerishableTwoGymnax(**env_kwargs),
            RSPerishableTwoGymnax().default_params,
        )
    elif env_name == "RSPerishableOneGymnax":
        return (
            RSPerishableOneGymnax(**env_kwargs),
            RSPerishableOneGymnax().default_params,
        )
    elif env_name == "DeMoorPerishableGymnax":
        return (
            DeMoorPerishableGymnax(**env_kwargs),
            DeMoorPerishableGymnax().default_params,
        )
    elif env_name == "DeMoorPerishable":
        return (
            DeMoorPerishableMAJAX(**env_kwargs),
            DeMoorPerishableMAJAX().default_params,
        )
    elif env_name == "MirjaliliPerishablePlateletGymnax":
        return (
            MirjaliliPerishablePlateletGymnax(**env_kwargs),
            MirjaliliPerishablePlateletGymnax().default_params,
        )
    elif env_name == "SimpleTwoProductPerishableGymnax":
        return (
            SimpleTwoProductPerishableGymnax(**env_kwargs),
            SimpleTwoProductPerishableGymnax().default_params,
        )
    elif env_name == "SimpleTwoProductPerishable":
        return (
            SimpleTwoProductPerishableEnv(**env_kwargs),
            SimpleTwoProductPerishableEnv().default_params,
        )
    elif env_name == "SimpleTwoProductPerishableLimitDemand":
        return (
            SimpleTwoProductPerishableLimitDemandEnv(**env_kwargs),
            SimpleTwoProductPerishableLimitDemandEnv().default_params,
        )
    elif env_name == "SimpleTwoProductPerishableLimitDemandGymnax":
        return (
            SimpleTwoProductPerishableLimitDemandGymnax(**env_kwargs),
            SimpleTwoProductPerishableLimitDemandGymnax().default_params,
        )
    elif env_name == "SimpleTwoProductPerishableIncIssueGymnax":
        return (
            SimpleTwoProductPerishableIncIssueGymnax(**env_kwargs),
            SimpleTwoProductPerishableIncIssueGymnax().default_params,
        )
    else:
        raise ValueError(f"Unknown environment '{env_name}'")


gymnax.make = make


class GymnaxFitness(object):
    def __init__(
        self,
        env_name: str = "CartPole-v1",
        num_env_steps: Optional[int] = None,
        num_rollouts: int = 16,
        env_kwargs: dict = {},
        env_params: dict = {},
        test: bool = False,
        n_devices: Optional[int] = None,
        num_warmup_days: int = 0,  # TODO: Arguments from here down are new, may want to input another way
        gamma: float = 0.99,
    ):
        self.env_name = env_name
        self.num_rollouts = num_rollouts
        self.test = test

        try:
            import gymnax
        except ImportError:
            raise ImportError(
                "You need to install `gymnax` to use its fitness rollouts."
            )

        # Define the RL environment & replace default parameters if desired
        self.env, self.env_params = make(env_name, **env_kwargs)
        self.env_params = self.env_params.create_env_params(**env_params)

        if num_env_steps is None:
            self.num_env_steps = int(self.env_params.max_steps_in_episode)
        else:
            self.num_env_steps = int(num_env_steps)
        self.steps_per_member = self.num_env_steps * num_rollouts

        ## TODO: Rewrite to make dicts with entries for each agent
        # self.action_shape = self.env.num_actions
        # self.input_shape = self.env.observation_space(self.env_params).spaces['obs'].shape

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices

        # Keep track of total steps executed in environment - blank out, side effect when we JIT rollout
        # self.total_env_steps = 0

        # Warmup and discounting parameters - newly added and to decide best way to track
        self.num_warmup_days = num_warmup_days
        self.gamma = gamma

    def set_apply_fn(self, network_apply, carry_init=None):
        """Set the network forward function."""
        self.network = network_apply
        # Set rollout function based on model architecture
        if carry_init is not None:
            self.single_rollout = self.rollout_rnn
            self.carry_init = carry_init
        else:
            self.single_rollout = self.rollout_ffw
        self.rollout_repeats = jax.vmap(self.single_rollout, in_axes=(0, None))
        self.rollout_pop = jax.vmap(self.rollout_repeats, in_axes=(None, 0))
        # pmap over popmembers if > 1 device is available - otherwise pmap
        if self.n_devices > 1:
            self.rollout_map = self.rollout_pmap
            print(
                f"GymFitness: {self.n_devices} devices detected. Please make"
                " sure that the ES population size divides evenly across the"
                " number of devices to pmap/parallelize over."
            )
        else:
            self.rollout_map = self.rollout_pop

    # TODO Would need to adjust to account for infos
    def rollout_pmap(self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree):
        """Parallelize rollout across devices. Split keys/reshape correctly."""
        keys_pmap = jnp.tile(rng_input, (self.n_devices, 1, 1))
        rew_dev, steps_dev = jax.pmap(self.rollout_pop)(keys_pmap, policy_params)
        rew_re = rew_dev.reshape(-1, self.num_rollouts)
        steps_re = steps_dev.reshape(-1, self.num_rollouts)
        return rew_re, steps_re

    def rollout(self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree):
        """Placeholder fn call for rolling out a population for multi-evals."""
        rng_pop = jax.random.split(rng_input, self.num_rollouts)
        scores, cum_infos, kpis, masks = jax.jit(self.rollout_map)(
            rng_pop, policy_params
        )
        # Update total step counter using only transitions before termination
        # self.total_env_steps += masks.sum()
        return scores, cum_infos, kpis

    def rollout_ffw(self, rng_input: chex.PRNGKey, policy_params: chex.ArrayTree):
        """Rollout an episode with lax.scan."""
        # Reset the environment
        rng_reset, rng_episode = jax.random.split(rng_input)
        obs, state = self.env.reset(rng_reset, self.env_params)

        def warmup_step(state_input):
            # New method added to handle warmup
            obs, state, _, rng = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action = self.network(policy_params, obs, rng=rng_net)
            next_o, next_s, reward, done, info = self.env.step(
                rng_step, state, action, self.env_params
            )
            warmup_done = jax.lax.ge(state.step, self.num_warmup_days)
            # TODO: We can probably get rid of the next few lines when using the while loop
            state = jax.tree_map(
                lambda x, y: jnp.where(warmup_done, x, y), state, next_s
            )
            obs = jax.tree_map(lambda x, y: jnp.where(warmup_done, x, y), obs, next_o)
            carry = (obs, state, warmup_done, rng)
            return carry

        def cond_fn_warmup(carry):
            """Condition for warmup loop."""
            obs, state, warmup_done, rng = carry
            return jax.lax.bitwise_not(warmup_done)

        def policy_step(state_input):
            """lax.scan compatible step transition in jax env."""
            # TODO: We can probably simplify some of this when using the while loop
            (
                obs,
                state,
                policy_params,
                rng,
                cum_reward,
                cum_return,
                cum_info,
                valid_mask,
            ) = state_input
            rng, rng_step, rng_net = jax.random.split(rng, 3)
            action = self.network(policy_params, obs, rng=rng_net)
            next_o, next_s, reward, done, info = self.env.step(
                rng_step, state, action, self.env_params
            )
            new_cum_reward = cum_reward + (reward * valid_mask)
            new_cum_return = cum_return + (
                (reward * self.gamma ** jnp.clip((state.step), a_min=0)) * valid_mask
            )  #
            new_cum_info = {k: v + cum_info[k] for k, v in info.items()}
            new_valid_mask = valid_mask * (1 - done)
            carry = [
                next_o,
                next_s,
                policy_params,
                rng,
                new_cum_reward,
                new_cum_return,
                new_cum_info,
                new_valid_mask,
            ]
            return carry

        def cond_fn_policy_step(carry):
            """Condition for warmup loop."""
            (
                next_o,
                next_s,
                policy_params,
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
            cond_fn_warmup,
            warmup_step,
            (
                obs,
                state,
                False,
                rng_episode,
            ),
        )

        obs, state, rng_episode, warmup_done = carry_out
        rng_reset, rng_episode = jax.random.split(rng_input)
        state = self.env.end_of_warmup_reset(rng_reset, state, self.env_params)

        cum_reward = 0.0
        cum_return = 0.0
        cum_info = self.env.empty_infos
        valid_mask = 1

        # Scan over episode step loop
        carry_out = jax.lax.while_loop(
            cond_fn_policy_step,
            policy_step,
            [
                obs,
                state,
                policy_params,
                rng_episode,
                cum_reward,
                cum_return,
                cum_info,
                valid_mask,
            ],
        )
        # Return the sum of rewards accumulated by agent in episode rollout
        # ep_mask = scan_out
        cum_reward = carry_out[-4].squeeze()  # Not discounted, one per agent
        cum_return = carry_out[
            -3
        ].squeeze()  # Discounted, just for replenishment for now; update rollout if we want to use it
        cum_infos = carry_out[-2]
        kpis = self.env.calculate_kpis(cum_infos)
        kpis["mean_daily_reward"] = cum_reward / cum_infos["day_counter"]
        # This allows us to incorporate a penalty when KPIs are breached over the whole episode
        # Aim to use it to enforce constraints on service level and wastage suggested by Meneses et al (2021)
        # for example on expriries and service level
        target_breached_penalty = self.env.calculate_target_kpi_penalty(
            kpis, self.env_params
        )
        cum_return = cum_return + target_breached_penalty
        return cum_return, cum_infos, kpis, None  # jnp.array(ep_mask)
