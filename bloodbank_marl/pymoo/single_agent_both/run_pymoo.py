import wandb
import hydra
import omegaconf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import chex
from typing import Tuple, Union, Optional, List, Dict
import jax
from evosax import ParameterReshaper
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
import numpy as np
import jax.numpy as jnp
from pymoo.indicators.hv import Hypervolume
import pickle


def calc_hypervolume(F: np.array, metrics_to_opt: List) -> float:

    ideal = {"wastage_%": 0, "service_level_%": -100, "exact_match_%": -100}
    nadir = {"wastage_%": 100, "service_level_%": 0, "exact_match_%": 0}
    ref_point = {"wastage_%": 100, "service_level_%": 0, "exact_match_%": 0}

    ref_point = np.array([ref_point[metric] for metric in metrics_to_opt])
    ideal = np.array([ideal[metric] for metric in metrics_to_opt])
    nadir = np.array([nadir[metric] for metric in metrics_to_opt])

    metric = Hypervolume(
        ref_point=ref_point,
        norm_ref_point=True,
        zero_to_one=True,
        ideal=ideal,
        nadir=nadir,
    )
    hv = metric.do(F)
    return hv


class SimpleTwoProductPerishableMultiAgentProbelem(Problem):
    def __init__(
        self,
        train_evaluator,
        param_reshaper,
        scenario_seed,
        # Decide whether to include variable in the optimization problem
        opt=["wastage_%", "service_level_%", "exact_match_%"],
        # Control use of constraints with these three arguments
        min_service_level_pc=-1.0,
        max_wastage_pc=101.0,
        min_exact_match_pc=-1.0,
        xl=-5.0,
        xu=5.0,
    ):

        # The default level of the constaints should always be met
        super().__init__(
            n_var=param_reshaper.total_params,
            n_obj=len(opt),
            n_ieq_constr=3,
            xl=xl,
            xu=xu,
        )

        self.train_evaluator = train_evaluator  #
        self.param_reshaper = param_reshaper
        self.rng = jax.random.PRNGKey(scenario_seed)

        self.opt = opt

        self.min_service_level_pc = min_service_level_pc
        self.max_wastage_pc = max_wastage_pc
        self.min_exact_match_pc = min_exact_match_pc

    def _evaluate(self, x, out, *args, **kwargs):
        self.rng, rng_train = jax.random.split(self.rng)
        x = jnp.array(x)  # Convert to jnp.array
        reshaped_params = self.param_reshaper.reshape(x)
        fitness, cum_infos, kpis = self.train_evaluator.rollout(
            rng_train, reshaped_params
        )
        wastage_pc_fitness = jnp.nan_to_num(kpis["wastage_%"].mean(axis=-1), nan=100.0)
        service_level_pc_fitness = jnp.nan_to_num(
            -1.0 * kpis["service_level_%"].mean(axis=-1), 0.0
        )  # Multiply by -1 so minimization will maximize service level;
        exact_match_pc_fitness = jnp.nan_to_num(
            -1 * kpis["exact_match_%"].mean(axis=-1), nan=0.0
        )  # Multiply by -1 so minimization will maximize exact match percentage;

        wastage_constraint = wastage_pc_fitness - self.max_wastage_pc
        service_level_constraint = service_level_pc_fitness + self.min_service_level_pc
        exact_match_constraint = exact_match_pc_fitness + self.min_exact_match_pc

        out["F"] = []
        if "wastage_%" in self.opt:
            out["F"].append(np.array(wastage_pc_fitness))
        if "service_level_%" in self.opt:
            out["F"].append(np.array(service_level_pc_fitness))
        if "exact_match_%" in self.opt:
            out["F"].append(np.array(exact_match_pc_fitness))

        out["G"] = [
            np.array(wastage_constraint),
            np.array(service_level_constraint),
            np.array(exact_match_constraint),
        ]


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: omegaconf.DictConfig):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb_config["pymoo"]["problem"]["xl"] = hydra.utils.call(cfg.pymoo.problem.xl)

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    rng = jax.random.PRNGKey(cfg.pymoo.seed)
    rng, rng_rep, rng_issue = jax.random.split(rng, 3)

    policy_params = {}
    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    policy_params[0] = policy_rep.get_initial_params(rng_rep)

    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    policy_params[1] = policy_issue.get_initial_params(rng_rep)

    param_reshaper = ParameterReshaper(policy_params)

    train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
    train_evaluator.set_apply_fn(policy_rep.apply)
    train_evaluator.set_issuing_fn(policy_issue.apply)

    # TODO: Enable specifying problem in cofig by defining in a separate file
    problem = SimpleTwoProductPerishableMultiAgentProbelem(
        train_evaluator=train_evaluator,
        param_reshaper=param_reshaper,
        scenario_seed=cfg.pymoo.seed,
        **wandb_config["pymoo"][
            "problem"
        ],  # Use wandb config as has already been resolved
    )

    algorithm = hydra.utils.instantiate(cfg.pymoo.algorithm)
    termination = hydra.utils.call(cfg.pymoo.termination)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=cfg.pymoo.seed,
        save_history=True,
        verbose=True,
    )
    F = res.F

    # Put results into a df
    cols = []
    if "wastage_%" in problem.opt:
        cols.append("Wastage")
    if "service_level_%" in problem.opt:
        cols.append("Service Level")
    if "exact_match_%" in problem.opt:
        cols.append("Exact Match")
    df = pd.DataFrame(F, columns=cols)
    if "service_level_%" in problem.opt:
        df["Service Level"] = -1 * df["Service Level"]
    if "exact_match_%" in problem.opt:
        df["Exact Match"] = -1 * df["Exact Match"]

    log_to_wandb = {}

    # Save results as wandb table
    table = wandb.Table(dataframe=df)

    # Calculate and log hypervolume
    hv = calc_hypervolume(F, problem.opt)
    log_to_wandb["train/hypervolume"] = hv

    # Plot results
    if "wastage_%" in problem.opt and "service_level_%" in problem.opt:
        log_to_wandb["train/service_level_v_wastage"] = wandb.plot.scatter(
            table, "Wastage", "Service Level"
        )
    if "exact_match_%" in problem.opt and "service_level_%" in problem.opt:
        log_to_wandb["train/service_level_v_exact_match"] = wandb.plot.scatter(
            table, "Exact Match", "Service Level"
        )
    if "exact_match_%" in problem.opt and "wastage" in problem.opt:
        log_to_wandb["train/exact_match_v_wastage"] = wandb.plot.scatter(
            table, "Wastage", "Exact Match"
        )

    # Eval runs
    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_rep.apply)
    test_evaluator.set_issuing_fn(policy_issue.apply)

    # Get the params out
    x = res.X
    reshaped_params = param_reshaper.reshape(x)

    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, reshaped_params)
    eval_df = pd.DataFrame(
        {
            "Wastage": kpis["wastage_%"].mean(axis=-1),
            "Service Level": kpis["service_level_%"].mean(axis=-1),
            "Exact Match": kpis["exact_match_%"].mean(axis=-1),
        }
    )

    # Save results as wandb table
    eval_table = wandb.Table(dataframe=eval_df)

    # Plot results
    log_to_wandb["eva;/service_level_v_wastage"] = wandb.plot.scatter(
        eval_table, "Wastage", "Service Level"
    )
    log_to_wandb["eval/service_level_v_exact_match"] = wandb.plot.scatter(
        eval_table, "Exact Match", "Service Level"
    )
    log_to_wandb["eval/exact_match_v_wastage"] = wandb.plot.scatter(
        eval_table, "Wastage", "Exact Match"
    )

    eval_hv_input = np.array(
        [
            (
                -1 * kpis[metric].mean(axis=-1)
                if metric in ["service_level_%", "exact_match_%"]
                else kpis[metric].mean(axis=-1)
            )
            for metric in problem.opt
        ]
    ).T
    eval_hv = calc_hypervolume(eval_hv_input, problem.opt)
    log_to_wandb["eval/hypervolume"] = eval_hv

    # if we need KPIs for each eval episode, extract relevant KPIs and save them with pickle
    if cfg.evaluation.record_overall_metrics_per_eval_rollout:
        # Do some work hereto save, we'd only use this when doing eval with best hps
        kpis_to_save = problem.opt
        eval_kpis = {}
        for kpi in kpis_to_save:
            eval_kpis[kpi] = kpis[kpi]
        pickle.dump(eval_kpis, open(f"{wandb.run.dir}/eval_kpis.pkl", "wb"))

    wandb.log(log_to_wandb)


if __name__ == "__main__":
    main()
