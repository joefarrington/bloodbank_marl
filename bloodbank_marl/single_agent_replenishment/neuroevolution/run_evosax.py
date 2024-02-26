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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Enable logging
log = logging.getLogger(__name__)


def plot_policy(policy_rep, policy_params, params_label="mean_params"):
    # For simplicity, redefine here without incluing in_transit
    # because for the simple example we can plot there is no in-transit
    # ANd having it with no shape causes problems
    @struct.dataclass
    class EnvObs:
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
    action_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 121).reshape(121, 11)
    all_obs = EnvObs(stock=stock, in_transit=in_transit, action_mask=action_mask)

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
    wandb.log({f"rep/policy_plot/{params_label}": wandb.Image(rep_heatmap)})


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    rng = jax.random.PRNGKey(cfg.evosax.seed)
    rng, rng_rep = jax.random.split(rng, 2)
    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)

    if cfg.policies.pretrained.enable:
        cpm = hydra.utils.instantiate(cfg.policies.pretrained.checkpoint_manager)
        policy_params = cpm.restore(cfg.policies.pretrained.checkpoint_id)[
            "trained_params"
        ]
    else:
        policy_params = policy_rep.get_initial_params(rng_rep)

    param_reshaper = ParameterReshaper(policy_params)
    test_param_reshaper = ParameterReshaper(policy_params, n_devices=1)

    train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
    train_evaluator.set_apply_fn(policy_rep.apply)

    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_rep.apply)

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

    if cfg.policies.pretrained.enable:
        state = strategy.initialize(
            rng_state_init, init_mean=param_reshaper.flatten_single(policy_params)
        )
    else:
        state = strategy.initialize(rng_state_init)

    # Logger
    es_logging = hydra.utils.instantiate(
        cfg.evosax.logging, num_dims=param_reshaper.total_params
    )
    es_log = es_logging.initialize()

    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)
    types = cfg.environment.types

    # Rollout the pretrained policy to start with
    if cfg.policies.pretrained.enable:
        fitness, cum_infos, kpis = test_evaluator.rollout(
            rng_eval,
            jax.tree_util.tree_map(lambda x: x.reshape((1,) + x.shape), policy_params),
        )
        cum_returns = fitness.mean(axis=1)

        group_metrics = cfg.environment.vector_kpis_to_log
        overall_metrics = cfg.environment.scalar_kpis_to_log
        log_to_wandb = {}

        for idx, p in enumerate(["pretrained"]):
            # Store group metrics in dataframe for W&B Table if we have them
            if group_metrics is not None:
                df = pd.DataFrame()
                for m in group_metrics:
                    df = pd.concat(
                        [df, pd.DataFrame(kpis[m][idx].mean(axis=(0)).reshape(1, -1))],
                        axis=0,
                    )
                    df = pd.concat(
                        [
                            df,
                            pd.DataFrame(kpis[m][idx].std(axis=(0)).reshape(1, -1)),
                        ],
                        axis=0,
                    )
                df.columns = types
                row_labels = [
                    f"{m}_{x}" for m in group_metrics for x in ["mean", "std"]
                ]
                df.insert(loc=0, column="metric", value=row_labels)
                wandb.log({f"eval/{p}/group_metrics": wandb.Table(dataframe=df)})

            # Add aggregate metrics and return to dict to be logged to W&B
            if overall_metrics is not None:
                for m in overall_metrics:
                    log_to_wandb[f"eval/{p}/{m}_mean"] = kpis[m][idx].mean()
                    log_to_wandb[f"eval/{p}/{m}_std"] = kpis[m][idx].std()

            log_to_wandb[f"eval/{p}/return_mean"] = fitness.mean()
            log_to_wandb[f"eval/{p}/return_std"] = fitness.std()
        wandb.log(log_to_wandb)

    for gen in range(cfg.evosax.num_generations):
        # TODO Do we want to split out eval, or should eval always be on the same set of rollouts?
        # We might want two set of eval, one for early stopping and one for final eval
        # If we're comparing this and simopt we want to be consistent
        rng, rng_init, rng_ask, rng_train = jax.random.split(rng, 4)
        x, state = strategy.ask(rng_ask, state)
        reshaped_params = param_reshaper.reshape(x)
        fitness, cum_infos, kpis = train_evaluator.rollout(rng_train, reshaped_params)
        fitness = fitness.mean(axis=-1)
        fit_re = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state)
        es_log = es_logging.update(es_log, x, fitness)
        best_params = es_log["top_params"][0]
        mean_params = state.mean

        log.info(f"Generation: {gen}, Performance: {es_log['log_top_1'][gen]}")

        log_to_wandb = {
            "Generation": gen,
            "train/gen_1_return": es_log["log_gen_1"][gen],
            "train/gen_mean_return": es_log["log_gen_mean"][gen],
            "train/top_1_return": es_log["log_top_1"][gen],
        }

        if (gen % cfg.evosax.evaluate_every_k_gens == 0) or (
            gen == cfg.evosax.num_generations - 1
        ):
            x_test = jnp.stack([best_params, mean_params], axis=0)
            reshaped_test_params = test_param_reshaper.reshape(x_test)

            fitness, cum_infos, kpis = test_evaluator.rollout(
                rng_train, reshaped_test_params
            )
            cum_returns = fitness.mean(axis=1)

            group_metrics = cfg.environment.vector_kpis_to_log
            overall_metrics = cfg.environment.scalar_kpis_to_log

            for idx, p in enumerate(["top_1", "mean_params"]):
                # Store group metrics in dataframe for W&B Table if we have them
                if group_metrics is not None:
                    df = pd.DataFrame()
                    for m in group_metrics:
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(
                                    kpis[m][idx].mean(axis=(0)).reshape(1, -1)
                                ),
                            ],
                            axis=0,
                        )
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame(kpis[m][idx].std(axis=(0)).reshape(1, -1)),
                            ],
                            axis=0,
                        )
                    df.columns = types
                    row_labels = [
                        f"{m}_{x}" for m in group_metrics for x in ["mean", "std"]
                    ]
                    df.insert(loc=0, column="metric", value=row_labels)
                    wandb.log({f"eval/{p}/group_metrics": wandb.Table(dataframe=df)})

                # Add aggregate metrics and return to dict to be logged to W&B
                if overall_metrics is not None:
                    for m in overall_metrics:
                        log_to_wandb[f"eval/{p}/{m}_mean"] = kpis[m][idx].mean()
                        log_to_wandb[f"eval/{p}/{m}_std"] = kpis[m][idx].std()

                log_to_wandb[f"eval/{p}/return_mean"] = fitness[idx].mean()
                log_to_wandb[f"eval/{p}/return_std"] = fitness[idx].std()

        wandb.log(log_to_wandb)

    policy_params_mean = test_param_reshaper.reshape(jnp.array([mean_params]))
    policy_params_top_1 = test_param_reshaper.reshape(jnp.array([best_params]))
    if cfg["plot_policy"]:
        plot_policy(policy_rep, policy_params_mean, "mean_params")
        plot_policy(policy_rep, policy_params_top_1, "top_1")

    params_to_save = jnp.stack([best_params, mean_params], axis=0)
    reshaped_params_to_save = test_param_reshaper.reshape(params_to_save)

    ckpt = {
        "state": state,
        "best_params": jax.tree_util.tree_map(lambda x: x[0], reshaped_params_to_save),
        "mean_params": jax.tree_util.tree_map(lambda x: x[1], reshaped_params_to_save),
    }
    checkpoint_manager.save(gen, ckpt)


if __name__ == "__main__":
    main()
