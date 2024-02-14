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
    ordinal_categorical_cross_entropy_with_integer_labels,
    get_obs_de_moor_perishable,
    get_obs_rs_multiproduct,
)
from bloodbank_marl.utils.gymnax_fitness import make
from pathlib import Path


def calculate_loss(state, params, batch):
    data_input, labels = batch
    # Obtain the logits and predictions of the model for the input data
    predictions = state.apply_fn(params, data_input).astype(jnp.float32)
    # Calculate the loss and accuracy
    loss = optax.l2_loss(predictions, labels).mean()
    return loss


@jax.jit
def train_step(state, batch):
    # Gradient function
    grad_fn = jax.value_and_grad(
        calculate_loss,  # Function to calculate the loss
        argnums=1,  # Parameters are second argument of the function
        has_aux=False,  # Function has additional outputs, here accuracy
    )
    # Determine gradients for current model, parameters and batch
    loss, grads = grad_fn(state, state.params, batch)
    # Perform parameter update with gradients and optimizer
    state = state.apply_gradients(grads=grads)
    # Return state and any other value we might want
    return state, loss


def eval_step(state, rng, cfg, nn_policy):
    # Roll out the heuristic policy
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(nn_policy.apply)
    fitness, cum_infos, kpis = test_evaluator.rollout(
        rng,
        jax.tree_util.tree_map(lambda x: x.reshape((1,) + x.shape), state.params),
    )
    return kpis["mean_daily_reward"].mean()


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
    log_to_wandb = {}
    num_epochs = cfg.pretraining.num_epochs
    for epoch in tqdm(range(num_epochs)):
        losses = []
        for batch_idx, batch in enumerate(data_loader):
            state, loss = train_step(state, batch)
            losses.append(loss)
        log_to_wandb.update(
            {
                "epoch": epoch,
                "training/loss": np.mean(losses),
            },
        )
        if epoch % cfg.evaluation.eval_freq == 0:
            fitness = eval_step(
                state, jax.random.PRNGKey(cfg.evaluation.seed), cfg, nn_policy
            )
            performance_gap = (fitness - heuristic_fitness) * 100 / heuristic_fitness
            log_to_wandb.update(
                {"eval/mean_return": fitness, "eval/performance_gap_%": performance_gap}
            )
            if performance_gap < best_performance_gap:
                best_performance_gap = performance_gap
                # TODO Add this as an artifact to W&B
                checkpoint_manager.save(
                    0, {"state": state, "trained_params": state.params}
                )
        wandb.log(log_to_wandb)
    return state


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):

    # Initialize wandb and log the config
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)
    rng = jax.random.PRNGKey(cfg.seed)

    # Instantiate a heuristic policy for labelling, and the parameters
    # Do a rollout to log the performance of the heuristic policy

    print("a")
    heuristic_policy = hydra.utils.instantiate(cfg.heuristic.policy)
    heuristic_params = hydra.utils.instantiate(cfg.heuristic.params).reshape(
        1, -1, 1
    )  # Intended for S params for SRep policy with multiple products
    print(heuristic_params)
    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(heuristic_policy.apply)
    fitness, cum_infos, kpis = test_evaluator.rollout(
        jax.random.PRNGKey(cfg.evaluation.seed), heuristic_params
    )
    # NOTE here we're using mean daily reward instead of return
    # Easy to work with for this scenario
    heuristic_fitness = kpis["mean_daily_reward"].mean()
    log_to_wandb = {"heuristic/mean_return": heuristic_fitness.mean()}
    for k in cfg.environment.scalar_kpis_to_log:
        log_to_wandb.update({f"heuristic/{k}_mean": kpis[k].mean()})
    wandb.log(log_to_wandb)
    # Get the observations for pretraining
    all_obs = get_obs_rs_multiproduct(cfg)

    # Label the obervations
    rng, _rng = jax.random.split(rng)
    labels = heuristic_policy.apply(heuristic_params, all_obs, _rng)

    # Apply preprocessing to the observations so that it does not need to be repeated during training
    all_obs = hydra.utils.call(cfg.pretraining.preprocess_observations, all_obs)
    all_labels = hydra.utils.call(cfg.pretraining.preprocess_labels, labels)

    # Create a dataset from the observations and labels
    dataset = RepDataset(all_obs, all_labels)
    train_data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.pretraining.batch_size,
        shuffle=True,
        collate_fn=collate_fn_multi_label,
    )

    # Instantiate the NN model for training. Note that we do not do preprocessing here, as it is done in the dataset
    env, default_env_params = make(
        cfg.environment.env_name, **cfg.environment.env_kwargs
    )
    nn_model = hydra.utils.instantiate(
        cfg.pretraining.nn_model, n_actions=env.num_actions
    )
    rng, _rng = jax.random.split(rng)
    nn_params = nn_model.init(_rng, all_obs)

    # Instantiate the nn replenishment policy and change the apply function of test evaluator.
    # This does have preprocessing because we use it in the eval steps to roll out the policy
    # compare to heuristic
    nn_policy = hydra.utils.instantiate(cfg.evaluation.replenishment_policy)
    test_evaluator.set_apply_fn(nn_policy.apply)

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
