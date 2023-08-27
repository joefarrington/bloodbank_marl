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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    rng = jax.random.PRNGKey(cfg.evosax.seed)
    rng, rng_rep, rng_issue = jax.random.split(rng, 3)

    env_cfg = omegaconf.OmegaConf.to_container(cfg.environment, resolve=True)
    env, default_env_params = make(env_cfg["env_name"], **env_cfg["env_kwargs"])

    rep_net = hydra.utils.instantiate(
        cfg.policy.replenishment,
        n_actions=env_cfg["env_kwargs"]["max_order_quantity"] + 1,
    )
    rep_obs = jnp.zeros(env.observation_space(env_cfg["env_params"], 0).shape)
    rep_params = rep_net.init(rng_rep, rep_obs)

    issue_net = hydra.utils.instantiate(
        cfg.policy.issuing, n_actions=env_cfg["env_kwargs"]["max_useful_life"] + 1
    )
    issue_obs = jnp.zeros(env.observation_space(env_cfg["env_params"], 1).shape)
    issue_params = issue_net.init(rng_issue, issue_obs)

    policy_params = {"rep": rep_params, "issue": issue_params}

    def policy_rep(policy_params, obs, rng):
        return rep_net.apply(policy_params["rep"], obs, rng)

    def policy_issue(policy_params, obs, rng):
        return issue_net.apply(policy_params["issue"], obs, rng)

    policies = [policy_rep, policy_issue]
    policy_manager = hydra.utils.instantiate(
        cfg.policy.policy_manager, policies=policies
    )

    param_reshaper = ParameterReshaper(policy_params)

    train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
    train_evaluator.set_apply_fn(policy_manager.apply)

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
        rng, rng_init, rng_ask, rng_eval = jax.random.split(rng, 4)
        x, state = strategy.ask(rng_ask, state)
        reshaped_params = param_reshaper.reshape(x)
        fitness = train_evaluator.rollout(rng_eval, reshaped_params).mean(axis=-1)
        fit_re = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state)
        log = es_logging.update(log, x, fitness)
        best_params = log["top_params"][0]
        mean_params = state.mean

        wandb.log(
            {
                "Generation": gen,
                "gen_1": log["log_gen_1"][gen],
                "gen_mean": log["log_gen_mean"][gen],
                "top_1": log["log_top_1"][gen],
            }
        )
        ckpt = {
            "state": state,
            "best_params": param_reshaper.reshape(best_params.reshape(1, -1)),
            "mean_params": param_reshaper.reshape(mean_params.reshape(1, -1)),
        }
        checkpoint_manager.save(gen, ckpt)


if __name__ == "__main__":
    main()
