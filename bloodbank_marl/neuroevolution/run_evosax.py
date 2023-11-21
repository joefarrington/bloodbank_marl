import bloodbank_marl
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import DeMoorPerishableMAJAX
from bloodbank_marl.utils.gymnax_fitness import GymnaxFitness, make
from bloodbank_marl.policies.replenishment import FlaxRepPolicy
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

# TODO We could subclass the logger or find another way to log the KPIs (especially for the best current params)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    rng = jax.random.PRNGKey(cfg.evosax.seed)
    rng, rng_rep, rng_issue = jax.random.split(rng, 3)

    # TODO: Think about how best to flag up whether policy is to be optimized
    # For now, raise an error if no policies are to be optimized
    # THere's no point in running this script, we can direct to a pure eval script
    if not cfg.policies.optimize:
        raise ValueError("No policies to optimize")

    policy_params = {}
    # 0 is the id for replenishment
    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    if 0 in cfg.policies.optimize:
        rep_params = policy_rep.get_params(rng_rep)
        policy_params[0] = rep_params

    # 1 is the id for issuing policy
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    if 1 in cfg.policies.optimize:
        issue_params = policy_issue.get_params(rng_issue)
        policy_params[1] = issue_params

    policies = [policy_rep.apply, policy_issue.apply]
    policy_manager = hydra.utils.instantiate(
        cfg.policies.policy_manager, policies=policies
    )

    param_reshaper = ParameterReshaper(policy_params)
    test_param_reshaper = ParameterReshaper(policy_params, n_devices=1)

    train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
    train_evaluator.set_apply_fn(policy_manager.apply)

    test_evaluator = hydra.utils.instantiate(cfg.test_evaluator)
    test_evaluator.set_apply_fn(policy_manager.apply)

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
    log = es_logging.initialize()

    for gen in range(cfg.evosax.num_generations):
        rng, rng_init, rng_ask, rng_train, rng_eval = jax.random.split(rng, 5)
        x, state = strategy.ask(rng_ask, state)
        reshaped_params = param_reshaper.reshape(x)
        fitness = train_evaluator.rollout(rng_train, reshaped_params)[0].mean(axis=-1)
        fit_re = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state)
        log = es_logging.update(log, x, fitness)
        best_params = log["top_params"][0]
        mean_params = state.mean

        log_to_wandb = {
            "Generation": gen,
            "train/gen_1": log["log_gen_1"][gen],
            "train/gen_mean": log["log_gen_mean"][gen],
            "train/top_1": log["log_top_1"][gen],
        }

        if gen % cfg.evosax.evaluate_every_k_gens == 0:
            x_test = jnp.stack([best_params, mean_params], axis=0)
            reshaped_test_params = test_param_reshaper.reshape(x_test)
            fitness, cum_infos, kpis = test_evaluator.rollout(
                rng_eval, reshaped_test_params
            )
            test_fitness_mean = fitness.mean(axis=-1)
            test_fitness_std = fitness.std(axis=-1)
            # NOTE: Mean KPIs assume single value per rollout (vs others that
            # are by product type etc; so we specifiy in config which ones should
            # be used here)
            test_kpis = {
                k: v.mean(axis=-1)
                for k, v in kpis.items()
                if k in cfg.environment.kpis_log_eval
            }
            for idx, p in enumerate(["top_1", "mean_params"]):
                log_to_wandb[f"eval/{p}/return_mean"] = test_fitness_mean[idx]
                log_to_wandb[f"eval/{p}/return_std"] = test_fitness_std[idx]
                for k, v in test_kpis.items():
                    log_to_wandb[f"eval/{p}/{k}"] = v[idx]

        wandb.log(log_to_wandb)
    # NOTE: Training is quick so for now just checkpoint at the end and re-run if needed
    ckpt = {
        "state": state,
        "best_params": param_reshaper.reshape(best_params.reshape(1, -1)),
        "mean_params": param_reshaper.reshape(mean_params.reshape(1, -1)),
    }
    checkpoint_manager.save(gen, ckpt)


if __name__ == "__main__":
    main()
