"""
from bloodbank_marl.scenarios.de_moor_perishable.gymnax_env import (
    DeMoorPerishableGymnax,
)
from bloodbank_marl.scenarios.de_moor_perishable.jax_env import DeMoorPerishableMAJAX
from bloodbank_marl.scenarios.meneses_perishable.jax_env import MenesesPerishableEnv
from bloodbank_marl.scenarios.meneses_perishable.gymnax_env import (
    MenesesPerishableGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.gymnax_env import RSPerishableGymnax
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_four import (
    RSPerishableFourGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_two import (
    RSPerishableTwoGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_one import (
    RSPerishableOneGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.jax_env import RSPerishableEnv
from bloodbank_marl.scenarios.rs_perishable.jax_env_two import RSPerishableTwoEnv
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_try_issue_too import (
    RSPerishableIncIssueGymnax,
)
from bloodbank_marl.scenarios.mirjalili_perishable_platelet.gymnax_env import (
    MirjaliliPerishablePlateletGymnax,
)
from bloodbank_marl.scenarios.simple_two_product.gymnax_env import (
    SimpleTwoProductPerishableGymnax,
)
from bloodbank_marl.scenarios.simple_two_product.jax_env import (
    SimpleTwoProductPerishableEnv,
)
from bloodbank_marl.scenarios.simple_two_product.jax_env_limit_demand import (
    SimpleTwoProductPerishableLimitDemandEnv,
)
from bloodbank_marl.scenarios.simple_two_product.gymnax_env_limit_demand import (
    SimpleTwoProductPerishableLimitDemandGymnax,
)
from bloodbank_marl.scenarios.simple_two_product.gymnax_env_try_issue_too import (
    SimpleTwoProductPerishableIncIssueGymnax,
)
from bloodbank_marl.scenarios.rs_perishable.gymnax_env_four_try_issue_too import (
    RSPerishableFourIncIssueGymnax,
)


def make(env_name, **env_kwargs):
    if env_name == "MenesesPerishable":
        return (
            MenesesPerishableEnv(**env_kwargs),
            MenesesPerishableEnv().default_params,
        )
    elif env_name == "MenesesPerishableGymnax":
        return (
            MenesesPerishableGymnax(**env_kwargs),
            MenesesPerishableGymnax().default_params,
        )
    elif env_name == "RSPerishable":
        return (
            RSPerishableEnv(**env_kwargs),
            RSPerishableEnv().default_params,
        )

    elif env_name == "RSPerishableTwo":
        return (
            RSPerishableTwoEnv(**env_kwargs),
            RSPerishableTwoEnv().default_params,
        )

    elif env_name == "RSPerishableGymnax":
        return (
            RSPerishableGymnax(**env_kwargs),
            RSPerishableGymnax().default_params,
        )
    elif env_name == "RSPerishableIncIssueGymnax":
        return (
            RSPerishableIncIssueGymnax(**env_kwargs),
            RSPerishableIncIssueGymnax().default_params,
        )
    elif env_name == "RSPerishableFourIncIssueGymnax":
        return (
            RSPerishableFourIncIssueGymnax(**env_kwargs),
            RSPerishableFourIncIssueGymnax().default_params,
        )
    elif env_name == "RSPerishableFourGymnax":
        return (
            RSPerishableFourGymnax(**env_kwargs),
            RSPerishableFourGymnax().default_params,
        )
    elif env_name == "RSPerishableTwoGymnax":
        return (
            RSPerishableTwoGymnax(**env_kwargs),
            RSPerishableTwoGymnax().default_params,
        )
    elif env_name == "RSPerishableOneGymnax":
        return (
            RSPerishableOneGymnax(**env_kwargs),
            RSPerishableOneGymnax().default_params,
        )
    elif env_name == "DeMoorPerishableGymnax":
        return (
            DeMoorPerishableGymnax(**env_kwargs),
            DeMoorPerishableGymnax().default_params,
        )
    elif env_name == "DeMoorPerishable":
        return (
            DeMoorPerishableMAJAX(**env_kwargs),
            DeMoorPerishableMAJAX().default_params,
        )
    elif env_name == "MirjaliliPerishablePlateletGymnax":
        return (
            MirjaliliPerishablePlateletGymnax(**env_kwargs),
            MirjaliliPerishablePlateletGymnax().default_params,
        )
    elif env_name == "SimpleTwoProductPerishableGymnax":
        return (
            SimpleTwoProductPerishableGymnax(**env_kwargs),
            SimpleTwoProductPerishableGymnax().default_params,
        )
    elif env_name == "SimpleTwoProductPerishable":
        return (
            SimpleTwoProductPerishableEnv(**env_kwargs),
            SimpleTwoProductPerishableEnv().default_params,
        )
    elif env_name == "SimpleTwoProductPerishableLimitDemand":
        return (
            SimpleTwoProductPerishableLimitDemandEnv(**env_kwargs),
            SimpleTwoProductPerishableLimitDemandEnv().default_params,
        )
    elif env_name == "SimpleTwoProductPerishableLimitDemandGymnax":
        return (
            SimpleTwoProductPerishableLimitDemandGymnax(**env_kwargs),
            SimpleTwoProductPerishableLimitDemandGymnax().default_params,
        )
    elif env_name == "SimpleTwoProductPerishableIncIssueGymnax":
        return (
            SimpleTwoProductPerishableIncIssueGymnax(**env_kwargs),
            SimpleTwoProductPerishableIncIssueGymnax().default_params,
        )
    else:
        raise ValueError(f"Unknown environment '{env_name}'")
"""

from bloodbank_marl.scenarios.single_product_perishable.gymnax_env import (
    SingleProductPerishableGymnaxEnv,
)
from bloodbank_marl.scenarios.single_product_perishable.marl_env import (
    SingleProductPerishableMarlEnv,
)
from bloodbank_marl.scenarios.two_product_perishable.adapted_single_agent_env import (
    TwoProductPerishableAdaptedEnv,
)
from bloodbank_marl.scenarios.eight_product_perishable.adapted_single_agent_env import (
    EightProductPerishableAdaptedEnv,
)


def make(env_name, **env_kwargs):
    if env_name == "SingleProductPerishableGymnax":
        return (
            SingleProductPerishableGymnaxEnv(**env_kwargs),
            SingleProductPerishableGymnaxEnv().default_params,
        )
    elif env_name == "SingleProductPerishableMarl":
        return (
            SingleProductPerishableMarlEnv(**env_kwargs),
            SingleProductPerishableMarlEnv().default_params,
        )
    elif env_name == "TwoProductPerishableAdapted":
        return (
            TwoProductPerishableAdaptedEnv(**env_kwargs),
            TwoProductPerishableAdaptedEnv().default_params,
        )
    elif env_name == "EightProductPerishableAdapted":
        return (
            EightProductPerishableAdaptedEnv(**env_kwargs),
            EightProductPerishableAdaptedEnv().default_params,
        )
    else:
        raise ValueError(f"Unknown environment '{env_name}'")
