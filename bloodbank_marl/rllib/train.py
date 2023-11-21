import hydra
import ray
import omegaconf
from ray import tune
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from bloodbank_marl.rllib.callbacks.kpi_callbacks import DeMoorKpiCallback
from bloodbank_marl.rllib.models.register import register_custom_models
from ray.air.integrations.wandb import WandbLoggerCallback
from ray import air
import wandb


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(hydra_cfg):
    register_custom_models()
    example_env = hydra.utils.instantiate(hydra_cfg.env)

    def env_creator(hydra_cfg=None, config=None):
        return hydra.utils.instantiate(hydra_cfg["env"])

    example_env = env_creator(hydra_cfg)

    register_env(
        hydra_cfg.env_name, lambda config: PettingZooEnv(env_creator(hydra_cfg, config))
    )
    wandb_config = omegaconf.OmegaConf.to_container(
        hydra_cfg, resolve=True, throw_on_missing=True
    )

    config = (
        hydra.utils.instantiate(hydra_cfg.algorithm_base_config)
        .training(**hydra_cfg.algorithm_additional_config.training)
        .callbacks(hydra.utils.call(hydra_cfg.algorithm_additional_config.callbacks))
        .debugging(**hydra_cfg.algorithm_additional_config.debugging)
        .environment(**hydra_cfg.algorithm_additional_config.environment)
        .evaluation(**hydra_cfg.algorithm_additional_config.evaluation)
        .experimental(**hydra_cfg.algorithm_additional_config.experimental)
        .fault_tolerance(**hydra_cfg.algorithm_additional_config.experimental)
        .framework(hydra_cfg.algorithm_additional_config.framework)
        .multi_agent(
            policies={
                "replenishment": hydra.utils.instantiate(
                    hydra_cfg.algorithm_additional_config.multi_agent.policies.replenishment,
                    observation_space=example_env.observation_space("replenishment"),
                    action_space=example_env.action_space("replenishment"),
                    _convert_="partial",
                ),
                "issuing": hydra.utils.instantiate(
                    hydra_cfg.algorithm_additional_config.multi_agent.policies.issuing,
                    observation_space=example_env.observation_space("issuing"),
                    action_space=example_env.action_space("issuing"),
                    _convert_="partial",
                ),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=omegaconf.OmegaConf.to_container(
                hydra_cfg.algorithm_additional_config.multi_agent.policies_to_train
            ),
        )
        .offline_data(**hydra_cfg.algorithm_additional_config.offline_data)
        .python_environment(**hydra_cfg.algorithm_additional_config.python_environment)
        .reporting(**hydra_cfg.algorithm_additional_config.reporting)
        .resources(**hydra_cfg.algorithm_additional_config.resources)
        .rl_module(**hydra_cfg.algorithm_additional_config.rl_module)
        .rollouts(**hydra_cfg.algorithm_additional_config.rollouts)
    )

    results = tune.Tuner(
        hydra_cfg.tune.algorithm_name,
        param_space=config.to_dict(),
        run_config=hydra.utils.instantiate(
            hydra_cfg.tune.run_config,
            callbacks=[
                WandbLoggerCallback(
                    project=hydra_cfg.wandb.project, config=wandb_config
                )
            ],
            _convert_="partial",
        ),
    ).fit()


if __name__ == "__main__":
    main()
