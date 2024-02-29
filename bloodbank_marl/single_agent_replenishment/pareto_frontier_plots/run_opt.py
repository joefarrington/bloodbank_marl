import jax
import orbax
import wandb
import hydra
import pandas as pd
from evosax import ParameterReshaper
import numpy as np
import optuna
import chex
from typing import Optional, Dict
from optuna.study import Study
import omegaconf
from omegaconf.dictconfig import DictConfig
from bloodbank_marl.policies.common import HeuristicPolicy
from bloodbank_marl.utils.single_agent_gymnax_fitness import GymnaxFitness
import logging
import jax.numpy as jnp
from bloodbank_marl.single_agent_replenishment.simopt.run_simopt import (
    param_search_bounds_from_config,
    grid_search_space_from_config,
)

# Enable logging
log = logging.getLogger(__name__)

C = 1e8  # Large constant to use as a penalty for infeasible solutions


# Adapted from bloodbank_marl/single_agent_replenishment/run_simopt.py to include KPIs
def simopt_grid_sampler(
    cfg: DictConfig,
    policy: HeuristicPolicy,
    test_evaluator: GymnaxFitness,
    rng_eval: chex.PRNGKey,
    initial_policy_params: Optional[Dict[str, int]] = None,
) -> Study:
    """Run simulation optimization using Optuna's GridSampler to propose parameter values"""
    search_bounds = param_search_bounds_from_config(cfg, policy)
    search_space = grid_search_space_from_config(search_bounds, policy)
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, search_space=search_space, seed=cfg.param_search.seed
    )
    # NEW: Need to specify direction for each objective
    study = optuna.create_study(sampler=sampler, directions=["minimize", "maximize"])

    # If we have an initial set of policy params, enqueue a trial with those
    if initial_policy_params is not None:
        study.enqueue_trial(initial_policy_params)

    i = 1
    while (
        len(sampler._get_unvisited_grid_ids(study)) > 0
        and i <= cfg.param_search.max_iterations
    ):
        trials = []
        policy_params = []
        num_parallel_trials = min(
            len(sampler._get_unvisited_grid_ids(study)),
            cfg.param_search.max_parallel_trials,
        )
        for j in range(num_parallel_trials):
            trial = study.ask()
            trials.append(trial)
            policy_params.append(
                np.array(
                    [
                        trial.suggest_int(
                            f"{p}",
                            search_bounds[p]["low"],
                            search_bounds[p]["high"],
                        )
                        for p in policy.param_names.flat
                    ]
                ).reshape(policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)

        # NEW: CODE FOR KPIs
        wastage_pc = jnp.nan_to_num(kpis["wastage_%"].mean(axis=-1))
        service_level_pc = kpis["service_level_%"].mean(axis=-1)

        wastage_pc_std = jnp.nan_to_num(kpis["wastage_%"].std(axis=-1))
        service_level_pc_std = kpis["service_level_%"].std(axis=-1)

        for idx in range(num_parallel_trials):
            try:
                trials[idx].set_user_attr("wastage_pc_std", float(wastage_pc_std[idx]))
                trials[idx].set_user_attr(
                    "service_level_pc_std", float(service_level_pc_std[idx])
                )
                study.tell(
                    trials[idx], (float(wastage_pc[idx]), float(service_level_pc[idx]))
                )
            except RuntimeError:
                break
        i += 1
    return study


def run_grid_search_record_kpis(cfg: DictConfig):
    rep_policy = hydra.utils.instantiate(cfg.heuristic_policy)
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(rep_policy.apply)
    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)

    study = simopt_grid_sampler(cfg, rep_policy, test_evaluator, rng_eval, None)

    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(by="params_S", ascending=True)
    trials_df = trials_df.rename(
        columns={"values_0": "wastage_%_mean", "values_1": "service_level_%_mean"}
    )
    return trials_df


def run_neuro_opt_one_kpi(
    cfg: DictConfig,
    fitness_kpi_name: str,
    fitness_kpi_direction: str,
    penalty_kpi_name: str,
    penalty_kpi_limit: str,
) -> pd.DataFrame:
    rows = []
    best_params_last_round = None
    if penalty_kpi_name == "service_level_%":
        limit_range = hydra.utils.instantiate(cfg.kpi_ranges.service_level_pc)
    elif penalty_kpi_name == "wastage_%":
        limit_range = hydra.utils.instantiate(cfg.kpi_ranges.wastage_pc)
    else:
        raise ValueError("penalty_kpi_name must be 'service_level_pc' or 'wastage_pc'")

    if fitness_kpi_direction == "max":
        cfg.evosax.fitness_shaper.maximize = True
    elif fitness_kpi_direction == "min":
        cfg.evosax.fitness_shaper.maximize = False
    else:
        raise ValueError("fitness_kpi_direction must be 'max' or 'min'")

    for penalty_kpi_threshold in limit_range:

        rng = jax.random.PRNGKey(cfg.evosax.seed)
        rng, rng_rep = jax.random.split(rng, 2)
        policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)

        policy_params = policy_rep.get_initial_params(rng_rep)

        param_reshaper = ParameterReshaper(policy_params)
        test_param_reshaper = ParameterReshaper(policy_params, n_devices=1)

        train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
        train_evaluator.set_apply_fn(policy_rep.apply)

        test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
        test_evaluator.set_apply_fn(policy_rep.apply)

        # Strategy and fitness shaper

        strategy = hydra.utils.instantiate(
            cfg.evosax.strategy, num_dims=param_reshaper.total_params
        )
        fitness_shaper = hydra.utils.instantiate(cfg.evosax.fitness_shaper)
        rng, rng_state_init = jax.random.split(rng, 2)

        # If this isn't the first policy to be optimized for this KPI,
        # use the previous best params as a starting point for optimization
        if cfg.evosax.init_prev_best == True and best_params_last_round is not None:
            state = strategy.initialize(
                rng_state_init,
                init_mean=param_reshaper.flatten_single(best_params_last_round),
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

        for gen in range(cfg.evosax.num_generations):
            rng, rng_init, rng_ask, rng_train = jax.random.split(rng, 4)
            x, state = strategy.ask(rng_ask, state)
            reshaped_params = param_reshaper.reshape(x)
            fitness, cum_infos, kpis = train_evaluator.rollout(
                rng_train, reshaped_params
            )

            # New, use the KPIs to get the fitness of a trial
            if penalty_kpi_limit == "max":
                fitness = (
                    kpis[fitness_kpi_name].mean(axis=-1)
                    - (kpis[penalty_kpi_name].mean(axis=-1) > penalty_kpi_threshold) * C
                )
            elif penalty_kpi_limit == "min":
                fitness = (
                    kpis[fitness_kpi_name].mean(axis=-1)
                    + (kpis[penalty_kpi_name].mean(axis=-1) < penalty_kpi_threshold) * C
                )
            else:
                raise ValueError("penalty_kpi_limit must be 'max' or 'min'")

            fit_re = fitness_shaper.apply(x, fitness)

            state = strategy.tell(x, fit_re, state)
            es_log = es_logging.update(es_log, x, fitness)
            best_params = es_log["top_params"][0]
            mean_params = state.mean

        store = {}
        x_test = jnp.stack([best_params, mean_params], axis=0)
        reshaped_test_params = test_param_reshaper.reshape(x_test)

        fitness, cum_infos, kpis = test_evaluator.rollout(
            rng_eval, reshaped_test_params
        )
        cum_returns = fitness.mean(axis=1)

        group_metrics = cfg.environment.vector_kpis_to_log
        overall_metrics = cfg.environment.scalar_kpis_to_log

        for idx, p in enumerate(["top_1", "mean_params"]):
            if overall_metrics is not None:
                for m in overall_metrics:
                    store[f"eval/{p}/{m}_mean"] = kpis[m][idx].mean()
                    store[f"eval/{p}/{m}_std"] = kpis[m][idx].std()

            store[f"eval/{p}/return_mean"] = fitness[idx].mean()
            store[f"eval/{p}/return_std"] = fitness[idx].std()

        # TODO: Consider if we need to hardcode the names of the KPIs
        row = [
            penalty_kpi_threshold,
            float(store["eval/top_1/wastage_%_mean"]),
            float(store["eval/top_1/service_level_%_mean"]),
            float(store["eval/mean_params/wastage_%_mean"]),
            float(store["eval/mean_params/service_level_%_mean"]),
        ]
        rows.append(row)
        best_params_last_round = best_params
    res_df = pd.DataFrame(
        rows,
        columns=[
            "penalty_kpi_threshold",
            "top_1_wastage_%_mean",
            "top_1_service_level_%_mean",
            "mean_params_wastage_%_mean",
            "mean_params_service_level_%_mean",
        ],
    )
    return res_df


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    log.info("Starting grid search over heuristic policy parameters")
    # Run grid search over the possible heuristic order up to parameters
    heuristic_df = run_grid_search_record_kpis(cfg)
    heuristic_df.to_csv("heuristic_df.csv")
    wandb.log({f"heuristic": wandb.Table(dataframe=heuristic_df)})
    log.info("Grid search over heuristic policy parameters complete")

    log.info("Starting optimization of service level with maximum wastage")
    # Run neuroevo to optimize service level, with a maximum wastage
    wastage_limit_df = run_neuro_opt_one_kpi(
        cfg, "service_level_%", "max", "wastage_%", "max"
    )
    wastage_limit_df.to_csv("wastage_limit_df.csv")
    wandb.log({f"neurevo_wastage_limit": wandb.Table(dataframe=wastage_limit_df)})
    log.info("Optimization of service level with maximum wastage complete")

    log.info("Starting optimization of wastage with minimum service level")
    # Run neuroevo to optimize wastage, with a minimum service level
    service_level_limit_df = run_neuro_opt_one_kpi(
        cfg, "wastage_%", "min", "service_level_%", "min"
    )
    service_level_limit_df.to_csv("service_level_limit_df.csv")
    wandb.log(
        {f"neurevo_service_level_limit": wandb.Table(dataframe=service_level_limit_df)}
    )
    log.info("Optimization of wastage with minimum service level complete")

    log.info("All optimization runs complete, results saved to csv for plotting")


if __name__ == "__main__":
    main()