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
from omegaconf import OmegaConf


def plot_policies(policy_rep, policy_issue, policy_params, params_label="mean_params"):
    # For simplicity, redefine here without incluing in_transit
    # because for the simple example we can plot there is no in-transit
    # ANd having it with no shape causes problems
    @struct.dataclass
    class EnvObs:
        agent_id: int  # Following Tianhou;
        stock: chex.Array
        in_transit: chex.Array
        action_mask: chex.Array

        @property
        def obs(self):
            return jnp.hstack([self.stock])

    stock = jnp.array([[i, j] for i in range(0, 11) for j in range(0, 11)])
    in_transit = jnp.array([1] * 121).reshape(121, 1)[
        :, 1:
    ]  # Doing this, should get the empty array we expect
    agent_id = jnp.array([1] * 121).reshape(121, 1)
    action_mask = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] * 121).reshape(121, 11)
    all_obs = EnvObs(
        stock=stock, agent_id=agent_id, action_mask=action_mask, in_transit=in_transit
    )

    rep_actions = jax.vmap(
        jax.vmap(policy_rep.apply_deterministic, in_axes=(0, None, None)),
        in_axes=(None, 0, None),
    )(policy_params, all_obs, jax.random.PRNGKey(1))
    rep_df = pd.DataFrame(
        {
            "age_1": all_obs.stock[:, 0],
            "age_2": all_obs.stock[:, 1],
            "action": rep_actions.reshape(-1),
        }
    )
    rep_df = rep_df.pivot(columns="age_2", index="age_1", values="action").sort_index(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    rep_heatmap = sns.heatmap(
        rep_df, annot=True, cmap="Greens_r", vmax=5, square=True, ax=ax
    )
    wandb.log({f"rep/policy_plot/{params_label}": wandb.Image(rep_heatmap)})

    # Different action mask for issuing
    action_mask = jnp.hstack(
        [jnp.ones((121, 1)), jnp.where(stock > 0, 1, 0), jnp.zeros((121, 8))]
    )
    all_obs = EnvObs(
        stock=stock, agent_id=agent_id, action_mask=action_mask, in_transit=in_transit
    )

    issue_actions = jax.vmap(
        jax.vmap(policy_issue.apply_deterministic, in_axes=(0, None, None)),
        in_axes=(None, 0, None),
    )(policy_params, all_obs, jax.random.PRNGKey(1))
    issue_df = pd.DataFrame(
        {
            "age_1": all_obs.stock[:, 0],
            "age_2": all_obs.stock[:, 1],
            "action": issue_actions.reshape(-1),
        }
    )
    issue_df = issue_df.pivot(
        columns="age_2", index="age_1", values="action"
    ).sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    issue_heatmap = sns.heatmap(
        issue_df, annot=True, cmap="Greens_r", vmax=5, square=True, ax=ax
    )
    wandb.log({f"issue/policy_plot/{params_label}": wandb.Image(issue_heatmap)})


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg):
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )

    run = wandb.init(**wandb_config["wandb"]["init"], config=wandb_config)

    rng = jax.random.PRNGKey(cfg.evosax.seed)
    rng, rng_rep, rng_issue = jax.random.split(rng, 3)

    # THere's no point in running this script, we can direct to a pure eval script
    if not cfg.policies.optimize:
        raise ValueError("No policies to optimize")

    policy_params = {}

    # 0 is the id for replenishment
    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    if 0 in cfg.policies.optimize:
        if cfg.policies.pretrained.replenishment.enable:
            rep_cpm = hydra.utils.instantiate(
                cfg.policies.pretrained.replenishment.checkpoint_manager
            )
            rep_checkpoint_id = rep_cpm.latest_step()
            rep_params = rep_cpm.restore(rep_checkpoint_id)["trained_params"]
        else:
            rep_params = policy_rep.get_initial_params(rng_rep)
        policy_params[0] = rep_params

    # 1 is the id for issuing policy
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    if 1 in cfg.policies.optimize:
        if cfg.policies.pretrained.issuing.enable:
            issue_cpm = hydra.utils.instantiate(
                cfg.policies.pretrained.issuing.checkpoint_manager
            )
            issue_checkpoint_id = issue_cpm.latest_step()
            issue_params = issue_cpm.restore(issue_checkpoint_id)["trained_params"]
        else:
            issue_params = policy_issue.get_initial_params(rng_issue)
        policy_params[1] = issue_params

    policies = [policy_rep.apply, policy_issue.apply]
    policy_manager = hydra.utils.instantiate(
        cfg.policies.policy_manager, policies=policies
    )

    param_reshaper = ParameterReshaper(policy_params)
    test_param_reshaper = ParameterReshaper(policy_params, n_devices=1)

    train_evaluator = hydra.utils.instantiate(cfg.train_evaluator)
    train_evaluator.set_apply_fn(policy_manager.apply)

    test_evaluator = hydra.utils.instantiate(cfg.evaluation.test_evaluator)
    test_evaluator.set_apply_fn(policy_manager.apply)

    # Checkpointing for NN policies
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_options = hydra.utils.instantiate(cfg.evosax.checkpoint_options)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(
        wandb.run.dir, orbax_checkpointer, checkpoint_options
    )

    rng_eval = jax.random.PRNGKey(cfg.evaluation.seed)

    # Start by rolling out the pretrained policy
    if (
        cfg.policies.pretrained.replenishment.enable
        or cfg.policies.pretrained.issuing.enable
    ):

        log_to_wandb = {}
        fitness, cum_infos, kpis = test_evaluator.rollout(
            rng_eval,
            jax.tree_util.tree_map(lambda x: x.reshape((1,) + x.shape), policy_params),
        )
        test_fitness_mean = fitness.mean(axis=-1)
        test_fitness_std = fitness.std(axis=-1)
        # Mean KPIs assume single value per rollout (vs others that
        # are by product type etc; so we specifiy in config which ones should
        # be used here)
        test_kpis = {
            k: v.mean(axis=-1)
            for k, v in kpis.items()
            if k in cfg.environment.kpis_log_eval
        }
        for idx, p in enumerate(["pretrained"]):
            log_to_wandb[f"eval/{p}/return_mean"] = test_fitness_mean[idx]
            log_to_wandb[f"eval/{p}/return_std"] = test_fitness_std[idx]
            for k, v in test_kpis.items():
                log_to_wandb[f"eval/{p}/{k}"] = v[idx]

        wandb.log(log_to_wandb)

        rng, rng_fitness = jax.random.split(rng)
        # And get fitness on the training eval to warm-start the process
        init_fitness, cum_infos, kpis = train_evaluator.rollout(
            rng_fitness,
            jax.tree_util.tree_map(lambda x: x.reshape((1,) + x.shape), policy_params),
        )
        init_fitness = init_fitness.mean(axis=-1)

    # Strategy and fitness shaper
    strategy = hydra.utils.instantiate(
        cfg.evosax.strategy, num_dims=param_reshaper.total_params
    )
    evo_params = strategy.params_strategy.replace(
        **hydra.utils.instantiate(cfg.evosax.evo_params)
    )
    fitness_shaper = hydra.utils.instantiate(cfg.evosax.fitness_shaper)

    # Logger
    es_logging = hydra.utils.instantiate(
        cfg.evosax.logging, num_dims=param_reshaper.total_params
    )
    log = es_logging.initialize()

    rng, rng_state_init = jax.random.split(rng, 2)
    if (
        cfg.policies.pretrained.replenishment.enable
        or cfg.policies.pretrained.issuing.enable
    ):
        state = strategy.initialize(
            rng_state_init,
            init_mean=param_reshaper.flatten_single(policy_params),
            init_fitness=init_fitness,
            params=evo_params,
        )
        # We want to put this combination into the log
        log = es_logging.update(
            log, param_reshaper.flatten_single(policy_params), init_fitness
        )
    else:
        state = strategy.initialize(rng_state_init, params=evo_params)

    for gen in range(cfg.evosax.num_generations):
        rng, rng_init, rng_ask, rng_train = jax.random.split(rng, 4)
        x, state = strategy.ask(rng_ask, state)
        reshaped_params = param_reshaper.reshape(x)
        fitness = train_evaluator.rollout(rng_train, reshaped_params)[0].mean(axis=-1)
        fit_re = fitness_shaper.apply(x, fitness)
        state = strategy.tell(x, fit_re, state)
        log = es_logging.update(log, x, fitness)
        best_params = log["top_params"][0]
        mean_params = state.mean

        log_to_wandb = {
            "Generation": gen,
            "train/gen_1": log["log_gen_1"][gen],
            "train/gen_mean": log["log_gen_mean"][gen],
            "train/top_1": log["log_top_1"][gen],
        }
        if (gen % cfg.evosax.evaluate_every_k_gens == 0) or (
            gen == cfg.evosax.num_generations - 1
        ):
            x_test = jnp.stack([best_params, mean_params], axis=0)
            reshaped_test_params = test_param_reshaper.reshape(x_test)
            fitness, cum_infos, kpis = test_evaluator.rollout(
                rng_eval, reshaped_test_params
            )
            test_fitness_mean = fitness.mean(axis=-1)
            test_fitness_std = fitness.std(axis=-1)
            # Mean KPIs assume single value per rollout (vs others that
            # are by product type etc; so we specifiy in config which ones should
            # be used here)
            test_kpis = {
                k: v.mean(axis=-1)
                for k, v in kpis.items()
                if k in cfg.environment.kpis_log_eval
            }
            for idx, p in enumerate(["top_1", "mean_params"]):
                log_to_wandb[f"eval/{p}/return_mean"] = test_fitness_mean[idx]
                log_to_wandb[f"eval/{p}/return_std"] = test_fitness_std[idx]
                for k, v in test_kpis.items():
                    log_to_wandb[f"eval/{p}/{k}"] = v[idx]

        wandb.log(log_to_wandb)

    policy_rep = hydra.utils.instantiate(cfg.policies.replenishment)
    policy_issue = hydra.utils.instantiate(cfg.policies.issuing)
    policy_params_mean = test_param_reshaper.reshape(jnp.array([mean_params]))
    policy_params_top_1 = test_param_reshaper.reshape(jnp.array([best_params]))
    if cfg["plot_policies"]:
        plot_policies(policy_rep, policy_issue, policy_params_mean, "mean_params")
        plot_policies(policy_rep, policy_issue, policy_params_top_1, "top_1")

    params_to_save = jnp.stack([best_params, mean_params], axis=0)
    reshaped_params_to_save = test_param_reshaper.reshape(params_to_save)

    ckpt = {"state": state, "all_params": reshaped_params_to_save}

    if 0 in cfg.policies.optimize:
        ckpt["rep_best_params"] = jax.tree_util.tree_map(
            lambda x: x[0], reshaped_params_to_save[0]
        )
        ckpt["rep_mean_params"] = jax.tree_util.tree_map(
            lambda x: x[1], reshaped_params_to_save[0]
        )
    if 1 in cfg.policies.optimize:
        ckpt["issue_best_params"] = jax.tree_util.tree_map(
            lambda x: x[0], reshaped_params_to_save[1]
        )
        ckpt["issue_mean_params"] = jax.tree_util.tree_map(
            lambda x: x[1], reshaped_params_to_save[1]
        )

    checkpoint_manager.save(gen, ckpt)


if __name__ == "__main__":
    main()
