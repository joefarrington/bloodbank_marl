# For now, we want to log a plot, and then also the distance between the best point and 100% SL, 0% wastage -> that's the area we're having trouble getting coverage of
# Later can amend to include eval vs training etc
# Can also deal with constraints by including a penalty etc
# Also think about seeding, especially inside the problem class

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


class SingleProductPerishableMultiAgentProbelem(Problem):
    def __init__(
        self,
        train_evaluator,
        param_reshaper,
        scenario_seed,
        # Decide whether to include variable in the optimization problem
        opt=["wastage", "service_level"],
        # Control use of constraints with these three arguments
        min_service_level_pc=-1.0,
        max_wastage_pc=101.0,
        xl=-5.0,
        xu=5.0,
    ):

        # The default level of the constaints should always be met
        super().__init__(
            n_var=param_reshaper.total_params,
            n_obj=len(opt),
            n_ieq_constr=2,
            xl=xl,
            xu=xu,
        )

        self.train_evaluator = train_evaluator  #
        self.param_reshaper = param_reshaper
        self.rng = jax.random.PRNGKey(scenario_seed)

        self.opt = opt

        self.min_service_level_pc = min_service_level_pc
        self.max_wastage_pc = max_wastage_pc

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

        wastage_constraint = wastage_pc_fitness - self.max_wastage_pc
        service_level_constraint = service_level_pc_fitness + self.min_service_level_pc

        out["F"] = []
        if "wastage" in self.opt:
            out["F"].append(np.array(wastage_pc_fitness))
        if "service_level" in self.opt:
            out["F"].append(np.array(service_level_pc_fitness))

        out["G"] = [
            np.array(wastage_constraint),
            np.array(service_level_constraint),
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

    policies = [policy_rep.apply, policy_issue.apply]
    policy_manager = hydra.utils.instantiate(
        cfg.policies.policy_manager, policies=policies
    )

    param_reshaper = ParameterReshaper(policy_params)

    train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
    train_evaluator.set_apply_fn(policy_manager.apply)

    # NOTE: Could enable specifying problem in cofig by defining in a separate file
    problem = SingleProductPerishableMultiAgentProbelem(
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

    df = pd.DataFrame(F, columns=["Wastage", "Service Level"])
    df["Service Level"] = -1 * df["Service Level"]

    log_to_wandb = {}

    # Save results as wandb table
    table = wandb.Table(dataframe=df)

    # Calculate and log hypervolume
    hv = hydra.utils.instantiate(cfg.pymoo.hypervolume).do(F)
    log_to_wandb["train/hypervolume"] = hv

    # Plot results
    log_to_wandb["train/service_level_v_wastage"] = wandb.plot.scatter(
        table, "Wastage", "Service Level"
    )

    # Eval runs
    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_manager.apply)

    # Get the params out
    x = res.X
    reshaped_params = param_reshaper.reshape(x)

    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, reshaped_params)
    eval_df = pd.DataFrame(
        {
            "Wastage": kpis["wastage_%"].mean(axis=-1),
            "Service Level": kpis["service_level_%"].mean(axis=-1),
        }
    )

    # Save results as wandb table
    eval_table = wandb.Table(dataframe=eval_df)

    # Plot results
    log_to_wandb["eval/service_level_v_wastage"] = wandb.plot.scatter(
        table, "Wastage", "Service Level"
    )

    # Calculate and log hypervolume
    eval_hv_input = np.array(
        [
            kpis["wastage_%"].mean(axis=-1),
            -1 * (kpis["service_level_%"].mean(axis=-1)),
        ]
    ).T
    eval_hv = hydra.utils.instantiate(cfg.pymoo.hypervolume).do(eval_hv_input)
    log_to_wandb["eval/hypervolume"] = eval_hv

    # if we need KPIs for each eval episode, extract relevant KPIs and save them with pickle
    if cfg.evaluation.record_overall_metrics_per_eval_rollout:
        # Do some work hereto save, we'd only use this when doing eval with best hps
        kpis_to_save = ["service_level_%", "wastage_%"]
        eval_kpis = {}
        for kpi in kpis_to_save:
            eval_kpis[kpi] = kpis[kpi]
        pickle.dump(eval_kpis, open(f"{wandb.run.dir}/eval_kpis.pkl", "wb"))

    wandb.log(log_to_wandb)


if __name__ == "__main__":
    main()
