import bloodbank_marl
from bloodbank_marl.policies.replenishment import SPolicy
import jax
import jax.numpy as jnp
import wandb
import hydra
import omegaconf
import numpy as np
import pandas as pd
import logging
from bloodbank_marl.utils.gymnax_fitness import make

# TODO: Had to hardcode no env_kwargs in issuing config to avoid circular dependency (ie.e issuing policy wants to know env_kwargs, but one env_kwargs is the issuing policy
# Need to think about how best to resolve this

# Enable logging
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)
    # TODO Avoid having to hardcode the product types here
    types = ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
    rep_policy = hydra.utils.instantiate(cfg.policies.replenishment)
    # TODO: Better way of inputting policy parameters
    rpp = hydra.utils.instantiate(cfg.policies.replenishment_policy_params)
    rep_policy_params = jnp.array([rpp[f"S_{t}"] for t in types]).reshape(-1, 1)
    rw = hydra.utils.instantiate(cfg.rollout_wrapper, model_forward=rep_policy.apply)

    rng = jax.random.PRNGKey(cfg.evaluation.seed)
    log.info(f"Running {cfg.evaluation.num_rollouts} evaluation rollouts.")

    batch_results = rw.batch_rollout(
        jax.random.split(rng, cfg.evaluation.num_rollouts), rep_policy_params
    )
    log.info(f"Calcuting metrics.")

    env, env_params = make(cfg.environment.env_name, **cfg.environment.env_params)
    kpis = jax.vmap(env.calculate_kpis)(batch_results["info"])

    # TODO Avoid having to hardcode the different metrics here
    group_metrics = [
        "mean_demand_by_pt_blood_group",
        "mean_order_by_product",
        "service_level_%_by_pt_blood_group",
        "expiries_%_by_product",
        "mean_holding_by_product",
        "mean_age_at_transfusion_by_pt_blood_group",
        "exact_match_%_by_pt_blood_group",
    ]

    overall_metrics = [
        "mean_total_order",
        "service_level_%",
        "expiries_%",
        "mean_holding",
        "exact_match_%",
        "mean_age_at_transfusion",
        "unmet_demand_units",
        "expired_units",
    ]

    # Create a dataframe of KPIs by type and log to W&B as a table
    df = pd.DataFrame()
    for m in group_metrics:
        df = pd.concat([df, pd.DataFrame(kpis[m].mean(axis=0).reshape(1, -1))], axis=0)
        df = pd.concat([df, pd.DataFrame(kpis[m].std(axis=0).reshape(1, -1))], axis=0)
    df.columns = types
    row_labels = [f"{m}_{x}" for m in group_metrics for x in ["mean", "std"]]
    df.insert(loc=0, column="metric", value=row_labels)
    wandb.log({"group_metrics": wandb.Table(dataframe=df)})

    # Log aggregate metrics to W&B, plus return
    for m in overall_metrics:
        wandb.run.summary[f"{m}_mean"] = kpis[m].mean()
        wandb.run.summary[f"{m}_std"] = kpis[m].std()
    wandb.run.summary["return_mean"] = batch_results["cum_return"].mean()
    wandb.run.summary["return_std"] = batch_results["cum_return"].std()
    log.info(f"Done.")


if __name__ == "__main__":
    main()
