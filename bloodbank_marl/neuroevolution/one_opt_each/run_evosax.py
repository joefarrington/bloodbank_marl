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
    rep_params = policy_rep.get_params(rng_rep)
    policy_params[0] = rep_params

    # 1 is the id for issuing policy
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    issue_params = policy_issue.get_params(rng_issue)
    policy_params[1] = issue_params

    policies = [policy_rep.apply, policy_issue.apply]
    policy_manager = hydra.utils.instantiate(
        cfg.policies.policy_manager, policies=policies
    )

    rep_param_reshaper = ParameterReshaper(policy_params[0])
    issue_param_reshaper = ParameterReshaper(policy_params[1])
    train_param_reshaper = ParameterReshaper(policy_params, n_devices=1)
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
    rep_strategy = hydra.utils.instantiate(
        cfg.evosax.replenishment.strategy, num_dims=rep_param_reshaper.total_params
    )
    issue_strategy = hydra.utils.instantiate(cfg.evosax.issuing.strategy, num_dims=issue_param_reshaper.total_params)
    fitness_shaper = hydra.utils.instantiate(cfg.evosax.fitness_shaper)
    rng, rng_state_rep_init, rng_state_issue_init = jax.random.split(rng, 3)
    rep_state = rep_strategy.initialize(rng_state_rep_init)
    issue_state = issue_strategy.initialize(rng_state_issue_init)

    # TODO Logger
    es_logging = hydra.utils.instantiate(
        cfg.evosax.logging, num_dims=train_param_reshaper.total_params
    )
    log = es_logging.initialize()

    for gen in range(cfg.evosax.num_generations):
        rng, rng_init, rng_ask_rep, rng_ask_issue, rng_train, rng_eval = jax.random.split(rng, 6)
        rep_x, rep_state = rep_strategy.ask(rng_ask_rep, rep_state)
        issue_x, issue_state = issue_strategy.ask(rng_ask_issue, issue_state)
        x = jnp.hstack([rep_x, issue_x])
        reshaped_params = train_param_reshaper.reshape(x)
        fitness = train_evaluator.rollout(rng_train, reshaped_params)[0].mean(axis=-1)
        
        rep_fit_re = fitness_shaper.apply(rep_x, fitness)
        rep_state = rep_strategy.tell(rep_x, rep_fit_re, rep_state)

        issue_fit_re = fitness_shaper.apply(issue_x, fitness)
        issue_state = issue_strategy.tell(issue_x, issue_fit_re, issue_state)
        
        log = es_logging.update(log, x, fitness)
        best_params = log["top_params"][0]
        mean_params = jnp.hstack([rep_state.mean, issue_state.mean])

        log_to_wandb = {
            "Generation": gen,
            "gen_1": log["log_gen_1"][gen],
            "gen_mean": log["log_gen_mean"][gen],
            "top_1": log["log_top_1"][gen],
        }

        if gen % cfg.evosax.evaluate_every_k_gens == 0:
            x_test = jnp.stack([best_params, mean_params], axis=0)
            reshaped_test_params = test_param_reshaper.reshape(x_test)
            test_fitness = test_evaluator.rollout(rng_eval, reshaped_test_params)[0].mean(
                axis=-1
            )
            log_to_wandb["top_1_test"] = test_fitness[0]
            log_to_wandb["mean_params_test"] = test_fitness[1]

        wandb.log(log_to_wandb)
        # TODO Perhaps only update checkpoint when we're doing better on test fitness?
        ckpt = {
            "rep_state": rep_state,
            "issue_state": issue_state,

            "best_params": train_param_reshaper.reshape(best_params.reshape(1, -1)),
            "mean_params": train_param_reshaper.reshape(mean_params.reshape(1, -1)),
        }
        checkpoint_manager.save(gen, ckpt)


if __name__ == "__main__":
    main()
