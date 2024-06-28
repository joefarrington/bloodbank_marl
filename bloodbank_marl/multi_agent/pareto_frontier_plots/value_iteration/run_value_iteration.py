# Adapted from https://github.com/joefarrington/viso_jax/blob/main/viso_jax/value_iteration/run_value_iteration.py
from datetime import datetime
import logging
from jax.config import config as jax_config
import hydra
import jax
import pandas as pd
from omegaconf.dictconfig import DictConfig
from pathlib import Path
import wandb
import omegaconf
import jax.numpy as jnp
import numpy as np
import pickle

# Enable logging
log = logging.getLogger(__name__)


def run_vi_and_eval_one_cost_combination(cfg: DictConfig) -> DictConfig:
    """Run value iteration and evaluation for a single cost combination."""
    # Run value iteration
    VIR = hydra.utils.instantiate(cfg.vi_runner, output_directory=Path(wandb.run.dir))
    vi_output = VIR.run_value_iteration(**cfg.run_settings)

    shape = tuple(  # Include a leading batch dimension for compatavility with the test evaluator
        [
            cfg.environment.env_kwargs.max_order_quantity + 1
            for i in range(
                cfg.environment.env_kwargs.max_useful_life
                + cfg.environment.env_kwargs.lead_time
                - 1
            )
        ]
    )
    rep_params = jnp.array(vi_output["policy"].values).reshape(shape)
    # Run evaluation for identified policy
    policy_rep = hydra.utils.instantiate(
        cfg.policies.replenishment, fixed_policy_params=rep_params
    )
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)

    policies = [policy_rep.apply, policy_issue.apply]
    policy_manager = hydra.utils.instantiate(
        cfg.policies.policy_manager, policies=policies
    )

    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_manager.apply)

    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)

    policy_params = {
        0: jnp.array([[0]]),
        1: jnp.array([[0]]),
    }  # Just placeholders - we're using fixed params for replenishment and no params needed for FIFO/OUFO
    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)
    cum_returns = fitness.mean(axis=1)

    overall_metrics = cfg.environment.kpis_log_eval

    store = {}
    if overall_metrics is not None:
        for m in overall_metrics:
            store[f"eval/{m}_mean"] = kpis[m].mean()
            store[f"eval/{m}_std"] = kpis[m].std()

        store[f"eval/return_mean"] = fitness.mean()
        store[f"eval/return_std"] = fitness.std()
    row = [
        float(store["eval/service_level_%_mean"]),
        float(store["eval/wastage_%_mean"]),
        float(store["eval/return_mean"]),
    ]
    metrics_per_eval_rollout_one_policy = {}
    if cfg.evaluation.record_overall_metrics_per_eval_rollout:
        metrics_per_eval_rollout_one_policy = {
            m: kpis[m] for m in ["wastage_%", "service_level_%"]
        }
    return row, metrics_per_eval_rollout_one_policy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run value iteration using a range of cost combinations."""

    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    shortage_costs = hydra.utils.instantiate(cfg.shortage_costs)
    wastage_costs = hydra.utils.instantiate(cfg.wastage_costs)
    assert len(wastage_costs) == len(
        shortage_costs
    ), "The lengths of wastage_costs and shortage_costs should be the same so pairs can be evaluated"

    vi_df = pd.DataFrame(
        columns=[
            "wastage_cost",
            "shortage_cost",
            "service_level_%_mean",
            "wastage_%_mean",
            "mean_return",
        ],
    )

    # Run value iteration and evaluation for each cost combination
    rows = []
    metrics_per_eval_rollout = {m: None for m in ["wastage_%", "service_level_%"]}
    for w, s in zip(wastage_costs, shortage_costs):

        # Set costs in the config to current combinatiom
        cfg.environment.env_params.wastage_cost = float(w)
        cfg.environment.env_params.shortage_cost = float(s)

        row, metrics_per_eval_rollout_one_policy = run_vi_and_eval_one_cost_combination(
            cfg
        )
        row = np.array([w, s] + row).reshape(1, -1)
        vi_df = pd.concat([vi_df, pd.DataFrame(row, columns=vi_df.columns)])
        # Save after each iteration, because this can be time consuming
        vi_df.to_csv("vi_df.csv")

        if cfg.evaluation.record_overall_metrics_per_eval_rollout:
            for kpi_name in metrics_per_eval_rollout.keys():
                if metrics_per_eval_rollout[kpi_name] is None:
                    metrics_per_eval_rollout[kpi_name] = (
                        metrics_per_eval_rollout_one_policy[kpi_name]
                    )
                else:
                    metrics_per_eval_rollout[kpi_name] = np.vstack(
                        [
                            metrics_per_eval_rollout[kpi_name],
                            metrics_per_eval_rollout_one_policy[kpi_name],
                        ]
                    )
            # Save after each iteration, because this can be time consuming
            pickle.dump(
                metrics_per_eval_rollout,
                open(f"{wandb.run.dir}/eval_kpis.pkl", "wb"),
            )

    wandb.log({f"heuristic": wandb.Table(dataframe=vi_df)})


if __name__ == "__main__":
    main()