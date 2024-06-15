import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import logging
from datetime import datetime
import pandas as pd
import optuna
from typing import Dict, Tuple, List, Optional
from optuna.study import Study
import jax
import jax.numpy as jnp
import numpy as np
import gymnax
import chex
from bloodbank_marl.utils.yaml import to_yaml, from_yaml
from bloodbank_marl.policies.common import HeuristicPolicy
import wandb
from bloodbank_marl.utils.single_agent_gymnax_fitness import GymnaxFitness
from bloodbank_marl.utils.gymnax_fitness import make
import omegaconf
from pathlib import Pathi

# Enable logging
log = logging.getLogger(__name__)


def param_search_bounds_from_config(
    cfg: DictConfig, policy: HeuristicPolicy
) -> Dict[str, int]:
    """Create a dict of search bounds for each parameter from the conf
    g file"""
    # Specify search bounds for each parameter
    if cfg.param_search.search_bounds.all_params is None:
        try:
            search_bounds = {
                p: {
                    "low": cfg.param_search.search_bounds[p]["low"],
                    "high": cfg.param_search.search_bounds[p]["high"],
                }
                for p in policy.param_names.flat
            }
        except:
            raise ValueError(
                "Ranges for each parameter must be specified if not using same range for all parameters"
            )
    # Otherwise, use the same range for all parameters
    else:
        search_bounds = {
            p: {
                "low": cfg.param_search.search_bounds.all_params.low,
                "high": cfg.param_search.search_bounds.all_params.high,
            }
            for p in policy.param_names.flat
        }
    return search_bounds


def grid_search_space_from_config(
    search_bounds: Dict[str, int], policy: HeuristicPolicy
) -> Dict[str, List[int]]:
    """Create a grid search space from the search bounds"""
    search_space = {
        p: list(
            range(
                search_bounds[p]["low"],
                search_bounds[p]["high"] + 1,
            )
        )
        for p in policy.param_names.flat
    }
    return search_space


# Grid sampler is not straightforwardly compatible with the ask/tell
# interface so we need to treat it a bit differently to avoid
# to avoid duplication and handle RuntimeError
# https://github.com/optuna/optuna/issues/4121
def simopt_grid_sampler(
    cfg: DictConfig,
    policy: HeuristicPolicy,
    train_evaluator: GymnaxFitness,
    rng_fit: chex.PRNGKey,
    initial_policy_params: Optional[Dict[str, int]] = None,
) -> Study:
    """Run simulation optimization using Optuna's GridSampler to propose parameter values"""
    search_bounds = param_search_bounds_from_config(cfg, policy)
    search_space = grid_search_space_from_config(search_bounds, policy)
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, search_space=search_space, seed=cfg.param_search.seed
    )
    study = optuna.create_study(sampler=sampler, direction="maximize")

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
        log.info(f"Round {i}: Suggesting parameters")
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
        # Combine with a placeholder for the issuing policy params, which are not required here
        # because we only use heuristic issuing policies with SimOpt
        policy_params = {0: policy_params, 1: jnp.zeros((policy_params.shape[0], 1))}

        log.info(f"Round {i}: Simulating rollouts")
        fitness, cum_infos, kpis = train_evaluator.rollout(rng_fit, policy_params)

        log.info(f"Round {i}: Processing results")
        objectives = fitness.mean(axis=-1)

        for idx in range(num_parallel_trials):
            try:
                study.tell(trials[idx], objectives[idx])
            except RuntimeError:
                break
        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        i += 1
    return study


def simopt_other_sampler(
    cfg: DictConfig,
    policy: HeuristicPolicy,
    train_evaluator: GymnaxFitness,
    rng_fit: chex.PRNGKey,
    initial_policy_params: Optional[Dict[str, int]] = None,
) -> Study:
    """Run simulation optimization using an Optuna sampler other than GridSampler to propose parameter values"""
    search_bounds = param_search_bounds_from_config(cfg, policy)
    sampler = hydra.utils.instantiate(
        cfg.param_search.sampler, seed=cfg.param_search.seed
    )
    study = optuna.create_study(sampler=sampler, direction="maximize")

    # If we have an initial set of policy params, enqueue a trial with those
    if initial_policy_params is not None:
        study.enqueue_trial(initial_policy_params)

    # Counter for early stopping
    es_counter = 0

    for i in range(1, cfg.param_search.max_iterations + 1):
        trials = []
        policy_params = []
        log.info(f"Round {i}: Suggesting parameters")
        for j in range(cfg.param_search.max_parallel_trials):
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
        # Combine with a placeholder for the issuing policy params, which are not required here
        # because we only use heuristic issuing policies with SimOpt
        policy_params = {0: policy_params, 1: jnp.zeros((policy_params.shape[0], 1))}
        log.info(f"Round {i}: Simulating rollouts")
        fitness, cum_infos, kpis = train_evaluator.rollout(rng_fit, policy_params)
        log.info(f"Round {i}: Processing results")
        objectives = fitness.mean(axis=-1)

        for idx in range(cfg.param_search.max_parallel_trials):
            study.tell(trials[idx], objectives[idx])

        # NOTE Removed this now using gymnax fitness Override rollout_results; helps to avoid GPU OOM error on larger problems
        # rollout_results = 0
        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        best_params = np.array([v for v in study.best_params.values()]).reshape(
            policy.params_shape
        )
        # Perform early stopping starting on the second round
        if i > 1:
            if study.best_params == best_params_last_round:
                es_counter += 1
            else:
                es_counter = 0
        if es_counter >= cfg.param_search.early_stopping_rounds:
            log.info(
                f"No change in best parameters for {cfg.param_search.early_stopping_rounds} rounds. Stopping search."
            )
            break
        best_params_last_round = study.best_params
    return study


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run simulation optimization using Optuna to find the best parameters for a policy,
    and evaluate the policy using the best parameters on a separate set of rollouts"""
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    rep_policy = hydra.utils.instantiate(cfg.policies.replenishment)
    issue_policy = hydra.utils.instantiate(cfg.policies.issuing)
    train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
    train_evaluator.set_apply_fn(rep_policy.apply)
    train_evaluator.set_issuing_fn(issue_policy.apply)

    rng_fit = jax.random.PRNGKey(cfg.param_search.seed)

    # Initial policy params
    if cfg.policies.replenishment_policy_params is not None:
        initial_policy_params = omegaconf.OmegaConf.to_container(
            cfg.policies.replenishment_policy_params
        )
    else:
        initial_policy_params = None
    if cfg.param_search.sampler._target_ == "optuna.samplers.GridSampler":
        study = simopt_grid_sampler(
            cfg, rep_policy, train_evaluator, rng_fit, initial_policy_params
        )
    else:
        study = simopt_other_sampler(
            cfg, rep_policy, train_evaluator, rng_fit, initial_policy_params
        )

    trials_df = study.trials_dataframe()
    wandb.log({"optuna_trials": wandb.Table(dataframe=trials_df)})

    log.info(
        f"Simulation optimization complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
    )
    wandb.log(study.best_params)
    # Extract best params and add to output_info
    # We assume here that all parameters are integers
    # which they should be for the kinds of heuristic
    # policies we're using

    log.info("Running evaluation rollouts for the best params")

    best_params = np.array([v for v in study.best_params.values()]).reshape(
        rep_policy.params_shape
    )

    # Run evaluation for best policy, including computing kpis
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(rep_policy.apply)
    test_evaluator.set_issuing_fn(issue_policy.apply)
    # Policy params for 1 (issuing) is just a placeholder, we'll always use heuristic issuing policies with SimOpt
    # Putting into an array like because rollout expects a batch of policy parameters
    policy_params = {0: jnp.array([best_params]), 1: jnp.array([[0]])}
    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)
    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)

    group_metrics = cfg.environment.vector_kpis_to_log
    overall_metrics = cfg.environment.scalar_kpis_to_log

    # Log aggregate metrics to W&B, plus return
    if overall_metrics is not None:
        for m in overall_metrics:
            wandb.run.summary[f"eval/{m}_mean"] = kpis[m].mean()
            wandb.run.summary[f"eval/{m}_std"] = kpis[m].std()
    wandb.run.summary["eval/return_mean"] = fitness.mean()
    wandb.run.summary["eval/return_std"] = fitness.std()

    # Record the overall KPIs for the top 1 params for each eval rollout, for pairwise comparisons
    if cfg.evaluation.record_overall_metrics_per_eval_rollout:
        overall_metrics_per_eval_rollout_df = pd.DataFrame()
        for m in overall_metrics:
            overall_metrics_per_eval_rollout_df[m] = kpis[m][0]
        wandb.log(
            {
                f"eval/overall_metrics_per_eval_rollout": wandb.Table(
                    dataframe=overall_metrics_per_eval_rollout_df
                )
            }
        )

    types = cfg.environment.types

    # Create a dataframe of KPIs by type and log to W&B as a table
    if group_metrics is not None:
        group_metrics_df = pd.DataFrame()
        for m in group_metrics:
            group_metrics_df = pd.concat(
                [
                    group_metrics_df,
                    pd.DataFrame(kpis[m].mean(axis=(0, 1)).reshape(1, -1)),
                ],
                axis=0,
            )
            group_metrics_df = pd.concat(
                [
                    group_metrics_df,
                    pd.DataFrame(kpis[m].std(axis=(0, 1)).reshape(1, -1)),
                ],
                axis=0,
            )
        group_metrics_df.columns = types
        row_labels = [f"{m}_{x}" for m in group_metrics for x in ["mean", "std"]]
        group_metrics_df.insert(loc=0, column="metric", value=row_labels)
        wandb.log({"eval/group_metrics": wandb.Table(dataframe=group_metrics_df)})

    if "all_allocations" in kpis:
        allocations_df = pd.DataFrame(
            kpis["all_allocations"][0].mean(axis=(0)), columns=types
        )
        row_labels = types
        allocations_df.insert(loc=0, column="product", value=row_labels)
        wandb.log({"eval/all_allocations": wandb.Table(dataframe=allocations_df)})

    # Save the params for loading into other scripts
    jnp.save(Path(wandb.run.dir) / "best_params.npy", best_params)

    log.info(f"Done.")


if __name__ == "__main__":
    main()
