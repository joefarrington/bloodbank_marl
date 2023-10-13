import bloodbank_marl
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import DeMoorPerishableMAJAX
from bloodbank_marl.utils.gymnax_fitness import GymnaxFitness, make
from bloodbank_marl.heuristic_simopt.common import (
    param_search_bounds_from_config,
    grid_search_space_from_config,
    process_params_for_log,
    process_params_for_df,
    run_simopt,
    simopt_grid_sampler,
    simopt_other_sampler,
)
from bloodbank_marl.utils.yaml import to_yaml
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import checkpoints
from flax import struct
from typing import Tuple, Union, Optional
import chex
import orbax
import wandb
import hydra
import omegaconf
import numpy as np
import logging

# Enable logging
log = logging.getLogger(__name__)

# NOTE: For now we're assuming we would only want to use this SimOpt
# approach to fit heuristic replenishment policies. Not clear at this
# stage what a heuristic issuing policy would look like aside from
# exact match and preference order which we can already specify without
# any optimization.

# NOTE: Might need to think a bit more carefully about how we compare evosax
# and simopt, for evosax we currently use a test evaluator every k iterations
# which is different from our old simopt approach where we ran the whole thing until
# what we decided was convergence and then evaluated on a final set of rollouts.

# TODO We might want policy manager to store the full policy and then take the apply
# functions so that we don't need to pass rep policies separately into functions
# for running simopt. Wrap everything up a bit more neatly.


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    policies = [policy_rep.apply, policy_issue.apply]
    policy_manager = hydra.utils.instantiate(
        cfg.policies.policy_manager, policies=policies
    )

    train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
    train_evaluator.set_apply_fn(policy_manager.apply)

    test_evaluator = hydra.utils.instantiate(cfg.test_evaluator)
    test_evaluator.set_apply_fn(policy_manager.apply)

    study = run_simopt(cfg, train_evaluator, policy_rep, None)

    # TODO: Incorporate W&B, line up evaluation with evosax
    # Expecially in terms of seeding evaluation rollouts
    best_trial_idx = study.best_trial.number
    trials_df = study.trials_dataframe()
    wandb_trials_table = wandb.Table(dataframe=trials_df)
    run.log({"optuna_trials": wandb_trials_table})

    best_params = np.array([v for v in study.best_params.values()]).reshape(
        policy_rep.params_shape
    )
    output_info = {}
    output_info["policy_params"] = process_params_for_log(policy_rep, best_params)
    to_yaml(output_info, "best_params.yaml")
    wandb.log_artifact("./best_params.yaml")

    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)
    policy_params = jnp.array([best_params])
    scores, cum_info, kpis = test_evaluator.rollout(rng_eval, policy_params)
    log.info(f"Mean return on evaluation rollouts: {scores.mean()}")

    # Log the return, the KPIs and the best parameters to W&B
    wandb.log({"eval_mean_return": scores.mean()})
    wandb.log({k: v.mean() for k, v in kpis.items()})
    wandb.log(study.best_params)


if __name__ == "__main__":
    main()
