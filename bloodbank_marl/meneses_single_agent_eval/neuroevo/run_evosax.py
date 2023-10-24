import bloodbank_marl
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import DeMoorPerishableMAJAX
from bloodbank_marl.utils.gymnax_fitness import GymnaxFitness, make
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import checkpoints
from evosax import OpenES, PGPE, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
from flax import struct
from typing import Tuple, Union, Optional
import chex
import orbax
import wandb
import hydra
import omegaconf
from bloodbank_marl.scenarios.meneses_perishable.gymnax_env import (
    MenesesPerishableGymnax,
    EnvObs,
)
import logging

# Enable logging
log = logging.getLogger(__name__)

# TODO: Add rep policy as part of config


class FlaxRepPolicy:
    def __init__(
        self,
        policy_class,
        policy_kwargs,
        policy_id,
        env_name,
        env_kwargs={},
        env_params={},
    ):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.env_params = env_params
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.policy_net = policy_class(n_actions=env.n_products, **policy_kwargs)

    def get_params(self, rng):
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        # TODO We should really use obs shape for this
        _, state = env.reset(jax.random.PRNGKey(0), default_env_params)
        obs = EnvObs(stock=state.stock, in_transit=state.in_transit[:, 1:])
        flat_obs = obs.in_transit.sum(axis=-1) + obs.stock.sum(axis=-1)
        return self.policy_net.init(rng, flat_obs)

    def apply(self, policy_params, obs, rng):
        # Adjusted so doesn't need to specify policy_id when getting policy params
        flat_obs = obs.in_transit.sum(axis=-1) + obs.stock.sum(axis=-1)
        return self.policy_net.apply(policy_params, flat_obs, rng)


class RepMLP(nn.Module):
    n_hidden: int
    n_actions: int  # number of products
    max_order_quantity: int

    @nn.compact
    def __call__(self, x, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        total_order = jnp.argmax(
            nn.Dense(self.max_order_quantity)(x)
        )  # This bit sets the total order
        x = nn.Dense(self.n_actions)(x)
        x = x / x.sum(axis=-1)  # This bit sets the proportions
        s = jnp.round(x * total_order, 0).astype(jnp.int32)
        # X is total per product here
        return jnp.clip(s - x, a_min=0)


# TODO We could subclass the logger or find another way to log the KPIs (especially for the best current params)

# TODO For now we're using the single agent rollout manager instead of GymnaxFitness
# Easy to change, just need to change what is instantiated in the train_evaluator, the function used to get fitness,
# and set the apply function.


class FlaxRepPolicy:
    def __init__(
        self,
        policy_class,
        policy_kwargs,
        policy_id,
        env_name,
        env_kwargs={},
        env_params={},
    ):
        self.policy_id = policy_id
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.env_params = env_params
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        self.policy_net = policy_class(n_actions=env.n_products, **policy_kwargs)

    def get_params(self, rng):
        env, default_env_params = make(self.env_name, **self.env_kwargs)
        # TODO We should really use obs shape for this
        _, state = env.reset(jax.random.PRNGKey(0), default_env_params)
        obs = EnvObs(stock=state.stock, in_transit=state.in_transit[:, 1:])
        flat_obs = obs.in_transit.sum(axis=-1) + obs.stock.sum(axis=-1)
        return self.policy_net.init(rng, flat_obs)

    def apply(self, policy_params, obs, rng):
        # Adjusted so doesn't need to specify policy_id when getting policy params
        flat_obs = obs.in_transit.sum(axis=-1) + obs.stock.sum(axis=-1)
        return self.policy_net.apply(policy_params, flat_obs, rng)


class RepMLP(nn.Module):
    n_hidden: int
    n_actions: int  # number of products
    max_order_quantity: int

    @nn.compact
    def __call__(self, x, rng: Optional[chex.PRNGKey] = jax.random.PRNGKey(0)):
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_hidden)(x)
        x = nn.relu(x)
        total_order = jnp.argmax(
            nn.Dense(self.max_order_quantity)(x)
        )  # This bit sets the total order
        x = nn.Dense(self.n_actions)(x)
        x = x / x.sum(axis=-1)  # This bit sets the proportions
        s = jnp.round(x * total_order, 0).astype(jnp.int32)
        # X is total per product here
        return jnp.clip(s - x, a_min=0)
        return s


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    rng = jax.random.PRNGKey(cfg.evosax.seed)
    rng, rng_rep, rng_issue = jax.random.split(rng, 3)

    # policy_params = {}
    # 0 is the id for replenishment
    # policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    policy_rep = FlaxRepPolicy(
        RepMLP,
        {"n_hidden": 32, "max_order_quantity": 100},
        "rep",
        "MenesesPerishableGymnax",
    )
    rep_params = policy_rep.get_params(rng_rep)

    param_reshaper = ParameterReshaper(rep_params)
    test_param_reshaper = ParameterReshaper(rep_params, n_devices=1)

    train_evaluator = hydra.utils.instantiate(
        cfg.train_evaluator, model_forward=policy_rep.apply
    )

    test_evaluator = hydra.utils.instantiate(
        cfg.test_evaluator, model_forward=policy_rep.apply
    )

    # Checkpointing for NN policies
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_options = hydra.utils.instantiate(cfg.evosax.checkpoint_options)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        wandb.run.dir, orbax_checkpointer, checkpoint_options
    )

    # Strategy and fitness shaper
    strategy = hydra.utils.instantiate(
        cfg.evosax.strategy, num_dims=param_reshaper.total_params
    )
    fitness_shaper = hydra.utils.instantiate(cfg.evosax.fitness_shaper)
    rng, rng_state_init = jax.random.split(rng, 2)
    state = strategy.initialize(rng_state_init)

    # Logger
    es_logging = hydra.utils.instantiate(
        cfg.evosax.logging, num_dims=param_reshaper.total_params
    )
    es_log = es_logging.initialize()

    for gen in range(cfg.evosax.num_generations):
        # TODO Do we want to split out eval, or should eval always be on the same set of rollouts?
        # We might want two set of eval, one for early stopping and one for final eval
        # If we're comparing this and simopt we want to be consistent
        rng, rng_init, rng_ask, rng_train, rng_eval = jax.random.split(rng, 5)
        x, state = strategy.ask(rng_ask, state)
        reshaped_params = param_reshaper.reshape(x)
        batch_rng_train = jax.random.split(rng_train, cfg.evosax.num_train_rollouts)
        fitness = (
            train_evaluator.population_rollout_return_only(
                batch_rng_train, reshaped_params
            )["cum_return"]
            .mean(axis=1)
            .reshape(-1)
        )
        fit_re = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state)
        es_log = es_logging.update(es_log, x, fitness)
        best_params = es_log["top_params"][0]
        mean_params = state.mean

        log.info(f"Generation: {gen}, Performance: {es_log['log_top_1'][gen]}")

        log_to_wandb = {
            "Generation": gen,
            "gen_1_return": es_log["log_gen_1"][gen],
            "gen_mean_return": es_log["log_gen_mean"][gen],
            "top_1_return": es_log["log_top_1"][gen],
        }

        if gen % cfg.evosax.evaluate_every_k_gens == 0:
            x_test = jnp.stack([best_params, mean_params], axis=0)
            reshaped_test_params = test_param_reshaper.reshape(x_test)
            batch_rng_eval = jax.random.split(rng_eval, cfg.evosax.num_test_rollouts)
            test_rollout_results = test_evaluator.population_rollout(
                batch_rng_eval, reshaped_test_params
            )
            cum_returns = test_rollout_results["cum_return"].mean(axis=1)
            log_to_wandb["top_1_test_return"] = cum_returns[0]
            log_to_wandb["mean_params_test_return"] = cum_returns[1]

            log.info(f"Top 1 return on test evaluator{cum_returns[0]}")

            overall_metrics = [
                "mean_total_order",
                "service_level_%",
                "expiries_%",
                "mean_holding",
                "exact_match_%",
                "mean_age_at_transfusion",
                "unmet_demand_units",
                "expired_units",
            ]
            kpis = jax.vmap(jax.vmap((test_evaluator.env.calculate_kpis)))(
                test_rollout_results["info"]
            )
            for metric in overall_metrics:
                mean_metric = kpis[f"{metric}"].mean(axis=1)
                log_to_wandb[f"top_1_test_{metric}"] = mean_metric[0]
                log_to_wandb[f"mean_params_test_{metric}"] = mean_metric[1]

        wandb.log(log_to_wandb)
        # TODO Perhaps only update checkpoint when we're doing better on test fitness?
        # NOTE: For now, not using checkpoints
        # ckpt = {
        #    "state": state,
        #    "best_params": param_reshaper.reshape(best_params.reshape(1, -1)),
        #    "mean_params": param_reshaper.reshape(mean_params.reshape(1, -1)),
        # }
        # checkpoint_manager.save(gen, ckpt)


if __name__ == "__main__":
    main()
