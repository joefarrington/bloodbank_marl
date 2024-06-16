import bloodbank_marl
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


# NOTE: The heuristic policies aren't set up to take params in MA env, it was assumed they'd be fixed.
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
        cfg.policies.load_replenishment_policy_params, reshape_for_fitness=False
    )

    # 1 is the id for issuing policy
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    policy_params[1] = hydra.utils.call(
        cfg.policies.load_issuing_policy_params, reshape_for_fitness=False
    )

    # Add batch dimension to the policy for use with the test evaluator
    policy_params = jax.tree_map(lambda x: x.reshape((1,) + x.shape), policy_params)

    # If we have a PPO policy, we want to evaluate it deterministically
    rep_apply_fn = (
        policy_rep.apply_deterministic
        if hasattr(policy_rep, "apply_deterministic")
        else policy_rep.apply
    )
    issue_apply_fn = (
        policy_issue.apply_deterministic
        if hasattr(policy_issue, "apply_deterministic")
        else policy_issue.apply
    )
    policies = [rep_apply_fn, issue_apply_fn]
    policy_manager = hydra.utils.instantiate(
        cfg.policies.policy_manager, policies=policies
    )

    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_manager.apply)

    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)

    log_to_wandb = {}
    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)
    test_fitness_mean = fitness.mean(axis=-1)
    test_fitness_std = fitness.std(axis=-1)
    # NOTE: Mean KPIs assume single value per rollout (vs others that
    # are by product type etc; so we specifiy in config which ones should
    # be used here)
    test_kpis = {
        k: v.mean(axis=-1)
        for k, v in kpis.items()
        if k in cfg.environment.kpis_log_eval
    }

    # Record the overall KPIs for each eval rollout, for pairwise comparisons
    if cfg.evaluation.record_overall_metrics_per_eval_rollout:
        overall_metrics_per_eval_rollout_df = pd.DataFrame()
        for m in kpis.keys():
            overall_metrics_per_eval_rollout_df[m] = kpis[m][0]
        wandb.log(
            {
                f"eval/overall_metrics_per_eval_rollout": wandb.Table(
                    dataframe=overall_metrics_per_eval_rollout_df
                )
            }
        )

    log_to_wandb[f"eval/return_mean"] = test_fitness_mean
    log_to_wandb[f"eval/return_std"] = test_fitness_std
    for k, v in test_kpis.items():
        log_to_wandb[f"eval/{k}"] = v
    wandb.log(log_to_wandb)


if __name__ == "__main__":
    main()
