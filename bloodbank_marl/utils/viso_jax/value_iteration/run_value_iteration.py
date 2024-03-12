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

# Enable logging
log = logging.getLogger(__name__)


def run_vi_and_eval_one_cost_combination(cfg: DictConfig) -> DictConfig:
    """Run value iteration and evaluation for a single cost combination."""
    # Run value iteration
    VIR = hydra.utils.instantiate(cfg.vi_runner, output_directory=Path(wandb.run.dir))
    vi_output = VIR.run_value_iteration(**cfg.run_settings)

    # Run evaluation for identified policy
    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_rep.apply)
    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)

    shape = (
        1,
    ) + tuple(  # Include a leading batch dimension for compatavility with the test evaluator
        [
            cfg.environment.env_kwargs.max_order_quantity + 1
            for i in range(
                cfg.environment.env_kwargs.max_useful_life
                + cfg.environment.env_kwargs.lead_time
                - 1
            )
        ]
    )
    policy_params = jnp.array(vi_output["policy"].values).reshape(shape)

    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)
    cum_returns = fitness.mean(axis=1)

    overall_metrics = cfg.environment.scalar_kpis_to_log

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
        float(store["eval/mean_holding_mean"]),
        float(store["eval/return_mean"]),
    ]
    return row


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

    # Run value iteration and evaluation for each cost combination
    rows = []
    for w, s in zip(wastage_costs, shortage_costs):

        # Set costs in the config to current combinatiom
        cfg.environment.env_params.wastage_cost = float(w)
        cfg.environment.env_params.shortage_cost = float(s)

        print(cfg)
        row = [w, s] + run_vi_and_eval_one_cost_combination(cfg)
        rows.append(row)

    vi_df = pd.DataFrame(
        rows,
        columns=[
            "wastage_cost",
            "shortage_cost",
            "service_level_%_mean",
            "wastage_%_mean",
            "mean_holding_mean",
            "mean_return",
        ],
    )
    vi_df.to_csv("vi_df.csv")
    wandb.log({f"heuristic": wandb.Table(dataframe=vi_df)})


if __name__ == "__main__":
    main()
