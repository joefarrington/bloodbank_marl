import wandb
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state, checkpoints
from evosax import OpenES, PGPE, ParameterReshaper, FitnessShaper, NetworkMapper
from evosax.utils import ESLog
from flax import struct
from typing import Tuple, Union, Optional
import chex
import orbax
import optax
import wandb
import hydra
import omegaconf
import itertools
import torch
from tqdm.auto import tqdm
from bloodbank_marl.utils.pretraining import (
    RepDataset,
    collate_fn_pytree,
    collate_fn_single_label,
    collate_fn_multi_label,
    get_obs_de_moor_perishable,
    get_obs_rs_multiproduct,
    get_obs_rs_multiproduct_issuing,
)
from bloodbank_marl.utils.make_env import make
from pathlib import Path


def calculate_loss(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data
    predictions = state.apply_fn(params, data_input).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.softmax_cross_entropy_with_integer_labels(predictions, labels).mean()
    correct_preds = jnp.argmax(predictions, axis=-1) == labels
    accuracy = correct_preds.mean()
    num_incorrect_preds = len(correct_preds) - correct_preds.sum()
    return loss, (accuracy, num_incorrect_preds)


@jax.jit
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(
        calculate_loss,  # Function to calculate the loss
        argnums=1,  # Parameters are second argument of the function
        has_aux=True,  # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    (loss, metrics), grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    accuracy, num_incorrect_preds = metrics
    return state, loss, accuracy, num_incorrect_preds


def eval_step(state, rng, cfg, nn_policy):
    # Roll out the heuristic policy
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    policy_replenishment = hydra.utils.instantiate(cfg.policies.replenishment)
    test_evaluator.set_apply_fn(policy_replenishment.apply)
    test_evaluator.set_issuing_fn(nn_policy.apply)

    #
    policy_params = {
        0: jnp.array(
            [[0]]
        ),  # Placeholder, should be set using fixed policy params in config
        1: jax.tree_util.tree_map(lambda x: x.reshape((1,) + x.shape), state.params),
    }

    fitness, cum_infos, kpis = test_evaluator.rollout(
        rng,
        policy_params,
    )
    return kpis["mean_daily_reward"].mean(), {
        "eval/wastage": kpis["wastage_%"].mean(),
        "eval/service_level": kpis["service_level_%"].mean(),
        "eval/exact_match": kpis["exact_match_%"].mean(),
    }


def train_model(
    state,
    data_loader,
    checkpoint_manager,
    cfg,
    heuristic_fitness,
    nn_policy,
):
    # Training loop
    best_performance_gap = jnp.inf
    num_epochs = cfg.pretraining.num_epochs
    for epoch in tqdm(range(num_epochs)):
        losses = []
        accs = []
        total_incorrect_preds = 0
        log_to_wandb = {}
        for batch_idx, batch in enumerate(data_loader):
            state, loss, accuracy, num_incorrect_preds = train_step(state, batch)
            losses.append(loss)
            accs.append(accuracy)
            total_incorrect_preds += num_incorrect_preds
        log_to_wandb.update(
            {
                "epoch": epoch,
                "training/loss": np.mean(losses),
                "training/accuracy": np.mean(accs),
                "training/incorrect_preds": total_incorrect_preds,
            },
        )
        if (epoch % cfg.evaluation.eval_freq == 0) or (
            epoch == cfg.pretraining.num_epochs - 1
        ):
            fitness, kpis = eval_step(
                state, jax.random.PRNGKey(cfg.evaluation.seed), cfg, nn_policy
            )
            performance_gap = ((fitness - heuristic_fitness) * 100) / heuristic_fitness
            log_to_wandb.update(
                {
                    "eval/mean_daily_reward": fitness,
                    "eval/performance_gap_%": performance_gap,
                }
            )
            log_to_wandb.update(kpis)
            if performance_gap < best_performance_gap:
                best_performance_gap = performance_gap
                checkpoint_manager.save(
                    epoch, {"state": state, "trained_params": state.params}
                )
        wandb.log(log_to_wandb)
    return state


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    # We use two different types of environment here
    # The adapted single product environment is used for evaluation because it's faster
    # The full multi-product environment is used for collecting observations for pretraining
    # because it provies easier access to the issuing observations (this is obs_collection.environment in config)

    # Initialize wandb and log the config
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)
    rng = jax.random.PRNGKey(cfg.seed)

    # Instantiate a heuristic issuing policy for labelling, and the parameters
    # Do a rollout to log the performance of the heuristic issuing policy
    replenishment_policy = hydra.utils.instantiate(cfg.policies.replenishment)
    heuristic_issuing_policy = hydra.utils.instantiate(
        cfg.obs_collection.policies.issuing
    )
    policy_params = {
        0: jnp.array([[0]]),  # Placeholders, params set as fixed in config
        1: jnp.array([[0]]),
    }
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(replenishment_policy.apply)
    test_evaluator.set_issuing_fn(heuristic_issuing_policy.apply)
    fitness, cum_infos, kpis = test_evaluator.rollout(
        jax.random.PRNGKey(cfg.evaluation.seed), policy_params
    )
    # NOTE here we're using mean daily reward instead of return
    # Easy to work with for this scenario
    heuristic_fitness = kpis["mean_daily_reward"].mean()
    log_to_wandb = {"heuristic/mean_daily_reward": heuristic_fitness.mean()}
    for k in cfg.environment.scalar_kpis_to_log:
        log_to_wandb.update({f"heuristic/{k}_mean": kpis[k].mean()})
    wandb.log(log_to_wandb)

    # Get the observations for pretraining
    all_obs = get_obs_rs_multiproduct_issuing(cfg)

    # Label the obervations
    rng, _rng = jax.random.split(rng)
    labelling_policy = hydra.utils.instantiate(cfg.labelling_policy)
    apply_vmap = jax.vmap(labelling_policy.apply, in_axes=(None, 0, None))
    labels = apply_vmap(policy_params, all_obs, _rng)

    # Apply preprocessing to the observations so that it does not need to be repeated during training
    all_obs = hydra.utils.call(cfg.pretraining.preprocess_observations, all_obs)
    all_labels = hydra.utils.call(cfg.pretraining.preprocess_labels, labels)

    # Create a dataset from the observations and labels
    dataset = RepDataset(all_obs, all_labels)
    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.pretraining.batch_size,
        shuffle=True,
        collate_fn=collate_fn_single_label,  # We're using integer labels
    )

    # Instantiate the NN model for training. Note that we do not do preprocessing here, as it is done in the dataset
    collection_env, default_env_params = make(
        cfg.obs_collection.environment.env_name,
        **cfg.obs_collection.environment.env_kwargs,
    )
    nn_model = hydra.utils.instantiate(
        cfg.pretraining.nn_model,
        n_actions=cfg.environment.env_kwargs.n_products + 1,
    )
    rng, _rng = jax.random.split(rng)
    nn_params = nn_model.init(_rng, all_obs)

    # Instantiate the nn replenishment policy and change the apply function of test evaluator.
    # This does have preprocessing because we use it in the eval steps to roll out the policy
    # compare to heuristic
    nn_policy = hydra.utils.instantiate(cfg.policies.issuing)
    test_evaluator.set_apply_fn(replenishment_policy.apply)
    test_evaluator.set_issuing_fn(nn_policy.apply)

    # Instantiate the optimizer
    optimizer = hydra.utils.instantiate(cfg.pretraining.optimizer)

    # Instantiate the Flax training state
    training_state = train_state.TrainState.create(
        apply_fn=nn_model.apply, params=nn_params, tx=optimizer
    )

    # Instantiate the checkpoint manager
    checkpoint_manager = hydra.utils.instantiate(
        cfg.checkpoint_manager, directory=Path(wandb.run.dir) / "checkpoints"
    )

    # Train the policy. Evalaute every n epochs by running the policy in the environment. Save a checkpoint when
    # the gap between the heuristic policy and the trained policy has fallen between epochs
    training_state = train_model(
        training_state,
        train_data_loader,
        checkpoint_manager,
        cfg,
        heuristic_fitness,
        nn_policy,
    )


if __name__ == "__main__":
    main()
