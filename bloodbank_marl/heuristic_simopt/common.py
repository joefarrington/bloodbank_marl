# Adapted in part from https://github.com/joefarrington/plt_returns/utils/simopt.py

import hydra
from omegaconf.dictconfig import DictConfig
import optuna
from typing import Dict, Tuple, List, Optional
from optuna.study import Study
import jax.numpy as jnp
import numpy as np
import chex
from bloodbank_marl.policies.common import HeuristicPolicy
import logging
from math import inf
import jax
import pandas as pd
from bloodbank_marl.policies.policy_manager import PolicyManager
from bloodbank_marl.utils.gymnax_fitness import GymnaxFitness

# Enable logging
log = logging.getLogger(__name__)


# TODO: Update for this project
def param_search_bounds_from_config(
    cfg: DictConfig, policy: HeuristicPolicy
) -> Dict[str, int]:
    """Create a dict of search bounds for each parameter from the config file"""
    # Specify search bounds for each parameter
    if cfg.optuna.search_bounds.all_params is None:
        try:
            search_bounds = {
                p: {
                    "low": cfg.optuna.search_bounds[p]["low"],
                    "high": cfg.optuna.search_bounds[p]["high"],
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
                "low": cfg.optuna.search_bounds.all_params.low,
                "high": cfg.optuna.search_bounds.all_params.high,
            }
            for p in policy.param_names.flat
        }
    return search_bounds


# TODO: Update for new project
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


# TODO: Update for new project
def process_params_for_log(
    policy: HeuristicPolicy, params: chex.Array
) -> Dict[str, int]:
    """Process policy parameters for logging"""
    # If no row labels, we don't want a multi-level dict
    # so handle separately
    if policy.param_row_names == []:
        processed_params = {
            str(param_name): int(param_value)
            for param_name, param_value in zip(policy.param_names.flat, params.flat)
        }
    # If there are row labels, easiest to convert to a dataframe and then into nested dict
    else:
        processed_params = pd.DataFrame(
            params,
            index=policy.param_row_names,
            columns=policy.param_col_names,
        ).to_dict()
    return processed_params


# TODO: Update for new project
def process_params_for_df(
    policy: HeuristicPolicy, params: chex.Array
) -> Dict[str, int]:
    """Process policy parameters for adding to a dataframe"""
    return {
        str(param_name): int(param_value)
        for param_name, param_value in zip(policy.param_names.flat, params.flat)
    }


# TODO: Update for this project
def run_simopt(
    cfg: DictConfig,
    train_evaluator: GymnaxFitness,
    rep_policy: HeuristicPolicy,
    initial_params: Optional[Dict[str, int]] = None,
) -> Study:
    log.info(f"Starting simulation optimization")
    if cfg.optuna.sampler._target_ == "optuna.samplers.GridSampler":
        study = simopt_grid_sampler(cfg, train_evaluator, rep_policy, initial_params)
    else:
        study = simopt_other_sampler(cfg, train_evaluator, rep_policy, initial_params)

    log.info(
        f"Simulation optimization complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
    )
    return study


# Grid sampler is not straightforwardly compatible with the ask/tell
# interface so we need to treat it a bit differently to avoid
# to avoid duplication and handle RuntimeError
# https://github.com/optuna/optuna/issues/4121
# Done basic updating for new project
def simopt_grid_sampler(
    cfg: DictConfig,
    train_evaluator: GymnaxFitness,
    rep_policy: HeuristicPolicy,
    initial_policy_params: Optional[Dict[str, int]] = None,
) -> Study:
    """Run simulation optimization using Optuna's GridSampler to propose parameter values"""
    search_bounds = param_search_bounds_from_config(cfg, rep_policy)
    search_space = grid_search_space_from_config(search_bounds, rep_policy)
    sampler = hydra.utils.instantiate(
        cfg.optuna.sampler, search_space=search_space, seed=cfg.optuna.seed
    )
    study = optuna.create_study(sampler=sampler, direction="maximize")

    # If we have an initial set of policy params, enqueue a trial with those
    if initial_policy_params is not None:
        study.enqueue_trial(initial_policy_params)

    rng_eval = jax.random.PRNGKey(cfg.optuna.seed)

    i = 1
    while (
        len(sampler._get_unvisited_grid_ids(study)) > 0
        and i <= cfg.optuna.max_iterations
    ):
        trials = []
        policy_params = []
        log.info(f"Round {i}: Suggesting parameters")
        num_parallel_trials = min(
            len(sampler._get_unvisited_grid_ids(study)),
            cfg.optuna.max_parallel_trials,
        )
        print(num_parallel_trials)
        while len(policy_params) < num_parallel_trials:
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
                        for p in rep_policy.param_names.flat
                    ]
                ).reshape(rep_policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        log.info(f"Round {i}: Simulating rollouts")
        scores, cum_infos, kpis = train_evaluator.rollout(rng_eval, policy_params)
        log.info(f"Round {i}: Processing results")
        objectives = scores.mean(axis=(-1))
        for idx in range(num_parallel_trials):
            try:
                study.tell(trials[idx], objectives[idx])
            except RuntimeError:
                break
        # Override rollout_results; helps to avoid GPU OOM error on larger problems
        rollout_results = 0
        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        i += 1
    return study


# Done basic updating for new project
def simopt_other_sampler(
    cfg: DictConfig,
    train_evaluator: GymnaxFitness,
    rep_policy: HeuristicPolicy,
    initial_policy_params: Optional[Dict[str, int]] = None,
) -> Study:
    """Run simulation optimization using an Optuna sampler other than GridSampler to propose parameter values"""
    search_bounds = param_search_bounds_from_config(cfg, rep_policy)
    sampler = hydra.utils.instantiate(cfg.optuna.sampler, seed=cfg.optuna.seed)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    # If we have an initial set of policy params, enqueue a trial with those
    if initial_policy_params is not None:
        study.enqueue_trial(initial_policy_params)

    rng_eval = jax.random.PRNGKey(cfg.optuna.seed)

    # Counter for early stopping
    es_counter = 0

    for i in range(1, cfg.optuna.max_iterations + 1):
        trials = []
        policy_params = []
        log.info(f"Round {i}: Suggesting parameters")
        while len(policy_params) < cfg.optuna.max_parallel_trials:
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
                        for p in rep_policy.param_names.flat
                    ]
                ).reshape(rep_policy.params_shape)
            )
        policy_params = jnp.array(policy_params)
        log.info(f"Round {i}: Simulating rollouts")
        scores, cum_infos, kpis = train_evaluator.rollout(rng_eval, policy_params)
        log.info(f"Round {i}: Processing results")
        objectives = scores.mean(axis=(-1))

        for idx in range(cfg.optuna.max_parallel_trials):
            study.tell(trials[idx], objectives[idx])

        # Override rollout_results; helps to avoid GPU OOM error on larger problems
        rollout_results = 0
        log.info(
            f"Round {i} complete. Best params: {study.best_params}, mean return: {study.best_value:.4f}"
        )
        # Perform early stopping starting on the second round
        if i > 1:
            if study.best_params == best_params_last_round:
                es_counter += 1
            else:
                es_counter = 0
        if es_counter >= cfg.optuna.early_stopping_rounds:
            log.info(
                f"No change in best parameters for {cfg.optuna.early_stopping_rounds} rounds. Stopping search."
            )
            break
        best_params_last_round = study.best_params
    return study
