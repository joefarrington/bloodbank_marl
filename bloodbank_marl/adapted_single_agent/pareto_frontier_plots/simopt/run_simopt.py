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
from bloodbank_marl.utils.simopt import (
    param_search_bounds_from_config,
    grid_search_space_from_config,
)
import pickle

# Enable logging
log = logging.getLogger(__name__)

M = 1e10  # Large constant to use as a penalty for infeasible solutions


# Adapted from bloodbank_marl/single_agent_replenishment/run_simopt.py to include KPIs
def simopt_grid_sampler(
    cfg: DictConfig,
    policy: HeuristicPolicy,
    test_evaluator: GymnaxFitness,
    rng_eval: chex.PRNGKey,
    initial_policy_params: Optional[Dict[str, int]] = None,
) -> Study:
    """Run simulation optimization using Optuna's GridSampler to propose parameter values"""

    metrics_per_eval_rollout = {
        "params": None,
        "wastage_%": None,
        "service_level_%": None,
        "exact_match_%": None,
    }

    search_bounds = param_search_bounds_from_config(cfg, policy)
    search_space = grid_search_space_from_config(search_bounds, policy)
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, search_space=search_space, seed=cfg.param_search.seed
    )
    # NEW: Need to specify direction for each objective
    study = optuna.create_study(
        sampler=sampler, directions=["minimize", "maximize", "maximize"]
    )

    # If we have an initial set of policy params, enqueue a trial with those
    if initial_policy_params is not None:
        study.enqueue_trial(initial_policy_params)

    i = 1
    while (
        len(sampler._get_unvisited_grid_ids(study)) > 0
        and i <= cfg.param_search.max_iterations
    ):
        trials = []
        rep_params = []
        num_parallel_trials = min(
            len(sampler._get_unvisited_grid_ids(study)),
            cfg.param_search.max_parallel_trials,
        )
        for j in range(num_parallel_trials):
            trial = study.ask()
            trials.append(trial)
            rep_params.append(
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
        rep_params = jnp.array(rep_params)
        policy_params = {
            0: rep_params,
            1: jnp.zeros_like(rep_params),
        }  # Placeholder for issuing policy
        fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)

        # NEW: CODE FOR KPIs
        wastage_pc = jnp.nan_to_num(kpis["wastage_%"].mean(axis=-1))
        service_level_pc = kpis["service_level_%"].mean(axis=-1)
        exact_match_pc = kpis["exact_match_%"].mean(axis=-1)

        wastage_pc_std = jnp.nan_to_num(kpis["wastage_%"].std(axis=-1))
        service_level_pc_std = kpis["service_level_%"].std(axis=-1)
        exact_match_pc_std = kpis["exact_match_%"].std(axis=-1)

        for idx in range(num_parallel_trials):
            try:
                trials[idx].set_user_attr("wastage_pc_std", float(wastage_pc_std[idx]))
                trials[idx].set_user_attr(
                    "service_level_pc_std", float(service_level_pc_std[idx])
                ),
                trials[idx].set_user_attr(
                    "exact_match_pc_std", float(exact_match_pc_std[idx])
                ),
                study.tell(
                    trials[idx],
                    (
                        float(wastage_pc[idx]),
                        float(service_level_pc[idx]),
                        float(exact_match_pc[idx]),
                    ),
                )
            except RuntimeError:
                break
        if cfg.evaluation.record_overall_metrics_per_eval_rollout:
            # Save the KPIS
            for kpi_name in ["wastage_%", "service_level_%", "exact_match_%"]:
                if metrics_per_eval_rollout[kpi_name] is None:
                    metrics_per_eval_rollout[kpi_name] = kpis[kpi_name]
                else:
                    metrics_per_eval_rollout[kpi_name] = np.vstack(
                        [metrics_per_eval_rollout[kpi_name], kpis[kpi_name]]
                    )
            # And also the params so we can see which params gave which result
            if metrics_per_eval_rollout["params"] is None:
                metrics_per_eval_rollout["params"] = rep_params
            else:
                metrics_per_eval_rollout["params"] = np.vstack(
                    [metrics_per_eval_rollout["params"], rep_params]
                )
        i += 1
    return study, metrics_per_eval_rollout


def run_grid_search_record_kpis(cfg: DictConfig):
    policy_rep = hydra.utils.instantiate(cfg.heuristic_policies.replenishment)
    policy_issue = hydra.utils.instantiate(cfg.heuristic_policies.issuing)

    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_rep.apply)
    test_evaluator.set_issuing_fn(policy_issue.apply)
    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)

    study, metrics_per_eval_rollout = simopt_grid_sampler(
        cfg, policy_rep, test_evaluator, rng_eval, None
    )

    trials_df = study.trials_dataframe()
    trials_df = trials_df.rename(
        columns={
            "values_0": "wastage_%_mean",
            "values_1": "service_level_%_mean",
            "values_2": "exact_match_%_mean",
        }
    )
    return trials_df, metrics_per_eval_rollout


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    log.info("Starting grid search over heuristic policy parameters")
    # Run grid search over the possible heuristic order up to parameters for exact match issuing policy
    cfg.heuristic_policies.issuing._target_ = (
        "bloodbank_marl.policies.issuing.heuristic.ExactMatchIssuingPolicy"
    )
    heuristic_df, metrics_per_eval_rollout = run_grid_search_record_kpis(cfg)
    heuristic_df.to_csv("simopt_exact_match_df.csv")
    wandb.log({f"simopt_exact_match": wandb.Table(dataframe=heuristic_df)})
    if cfg.evaluation.record_overall_metrics_per_eval_rollout:
        pickle.dump(
            metrics_per_eval_rollout,
            open(f"{wandb.run.dir}/eval_kpis_exact_match.pkl", "wb"),
        )
    log.info(
        "Grid search over heuristic policy parameters with exact match issuing complete"
    )

    # Run grid search over the possible heuristic order up to parameters for priority match issuing policy
    cfg.heuristic_policies.issuing._target_ = (
        "bloodbank_marl.policies.issuing.heuristic.PriorityMatchIssuingPolicy"
    )
    heuristic_df, metrics_per_eval_rollout = run_grid_search_record_kpis(cfg)
    heuristic_df.to_csv("simopt_priority_match_df.csv")
    wandb.log({f"simopt_priority_match": wandb.Table(dataframe=heuristic_df)})
    if cfg.evaluation.record_overall_metrics_per_eval_rollout:
        pickle.dump(
            metrics_per_eval_rollout,
            open(f"{wandb.run.dir}/eval_kpis_priority_match.pkl", "wb"),
        )
    log.info(
        "Grid search over heuristic policy parameters with priority match issuing complete"
    )

    # Run grid search over the possible heuristic order up to parameters for oldest compatible match issuing policy
    cfg.heuristic_policies.issuing._target_ = (
        "bloodbank_marl.policies.issuing.heuristic.OldestCompatibleIssuingPolicy"
    )
    heuristic_df, metrics_per_eval_rollout = run_grid_search_record_kpis(cfg)
    heuristic_df.to_csv("simopt_oldest_compatible_match_df.csv")
    wandb.log({f"simopt_oldest_compatible_match": wandb.Table(dataframe=heuristic_df)})
    if cfg.evaluation.record_overall_metrics_per_eval_rollout:
        pickle.dump(
            metrics_per_eval_rollout,
            open(f"{wandb.run.dir}/eval_kpis_oldest_compatible_match.pkl", "wb"),
        )
    log.info(
        "Grid search over heuristic policy parameters with oldest compatible match issuing complete"
    )

    log.info("All optimization runs complete, results saved to csv for plotting")


if __name__ == "__main__":
    main()
