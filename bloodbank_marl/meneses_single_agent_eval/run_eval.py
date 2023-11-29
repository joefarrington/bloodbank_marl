import bloodbank_marl
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

    rep_policy = hydra.utils.instantiate(cfg.policies.replenishment)
    # TODO: Better way of inputting policy parameters
    types = ["O-", "O+", "A-", "A+", "B-", "B+", "AB-", "AB+"]
    rpp = hydra.utils.instantiate(cfg.policies.replenishment_policy_params)
    rep_policy_params = jnp.array([rpp[f"S_{t}"] for t in types]).reshape(-1, 1)

    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(rep_policy.apply)
    policy_params = jnp.array([rep_policy_params])
    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)
    log.info(
        f"Running {cfg.evaluation.test_evaluator.num_rollouts} evaluation rollouts."
    )
    fitness, cum_infos, kpis = test_evaluator.rollout(rng_eval, policy_params)

    # TODO Avoid having to hardcode the different metrics here
    group_metrics = cfg.environment.vector_kpis_to_log
    overall_metrics = cfg.environment.scalar_kpis_to_log

    # Create a dataframe of KPIs by type and log to W&B as a table
    df = pd.DataFrame()
    for m in group_metrics:
        df = pd.concat(
            [df, pd.DataFrame(kpis[m].mean(axis=(0, 1)).reshape(1, -1))], axis=0
        )
        df = pd.concat(
            [df, pd.DataFrame(kpis[m].std(axis=(0, 1)).reshape(1, -1))], axis=0
        )
    df.columns = types
    row_labels = [f"{m}_{x}" for m in group_metrics for x in ["mean", "std"]]
    df.insert(loc=0, column="metric", value=row_labels)
    wandb.log({"eval/group_metrics": wandb.Table(dataframe=df)})

    # Log aggregate metrics to W&B, plus return
    for m in overall_metrics:
        wandb.run.summary[f"{m}_mean"] = kpis[m].mean()
        wandb.run.summary[f"{m}_std"] = kpis[m].std()
    wandb.run.summary["eval/return_mean"] = fitness.mean()
    wandb.run.summary["eval/return_std"] = fitness.std()
    log.info(f"Done.")


if __name__ == "__main__":
    main()
