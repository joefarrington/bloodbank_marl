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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# TODO: The heuristic policies aren't set up to take params in MA env, it was assumed they'd be fixed.
# So, can load using similar function but provide to the fixed_params arg in the policy __init__ method.


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    policy_params = {}

    # 0 is the id for replenishment
    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    policy_params[0] = hydra.utils.call(
        cfg.policies.load_replenishment_policy_params,
        reshape_for_fitness=False,
    )

    # 1 is the id for issuing policy
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    policy_params[1] = hydra.utils.call(
        cfg.policies.load_issuing_policy_params,
        reshape_for_fitness=False,
    )

    # Add batch dimension to the policy for use with the test evaluator
    policy_params = jax.tree_map(lambda x: x.reshape((1,) + x.shape), policy_params)

    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_rep.apply)
    test_evaluator.set_issuing_fn(policy_issue.apply)

    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)

    log_to_wandb = {}
    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)

    group_metrics = cfg.environment.vector_kpis_to_log
    overall_metrics = cfg.environment.scalar_kpis_to_log
    types = cfg.environment.types

    # Add aggregate metrics and return to dict to be logged to W&B
    if overall_metrics is not None:
        for m in overall_metrics:
            log_to_wandb[f"eval/{m}_mean"] = kpis[m][0].mean()
            log_to_wandb[f"eval/{m}_std"] = kpis[m][0].std()

    log_to_wandb[f"eval/return_mean"] = fitness[0].mean()
    log_to_wandb[f"eval/return_std"] = fitness[0].std()
    wandb.log(log_to_wandb)

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

    if group_metrics is not None:
        df = pd.DataFrame()
        for m in group_metrics:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(kpis[m][0].mean(axis=(0)).reshape(1, -1)),
                ],
                axis=0,
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(kpis[m][0].std(axis=(0)).reshape(1, -1)),
                ],
                axis=0,
            )
        df.columns = types
        row_labels = [f"{m}_{x}" for m in group_metrics for x in ["mean", "std"]]
        df.insert(loc=0, column="metric", value=row_labels)
        wandb.log({f"eval/group_metrics": wandb.Table(dataframe=df)})

    if "all_allocations" in kpis:
        allocations_df = pd.DataFrame(
            kpis["all_allocations"][0].mean(axis=(0)), columns=types
        )
        row_labels = types
        allocations_df.insert(loc=0, column="product", value=row_labels)
        wandb.log({"eval/all_allocations": wandb.Table(dataframe=allocations_df)})


if __name__ == "__main__":
    main()
