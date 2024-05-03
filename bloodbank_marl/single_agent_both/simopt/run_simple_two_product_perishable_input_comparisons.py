import subprocess
from typing import List

heuristic_issuing_policy_targets = [
    "bloodbank_marl.policies.issuing.heuristic.ExactMatchIssuingPolicy",
    "bloodbank_marl.policies.issuing.heuristic.PriorityMatchIssuingPolicy",
    "bloodbank_marl.policies.issuing.heuristic.OldestCompatibleIssuingPolicy",
]

heuristic_issuing_policy_tags = [
    "exact_match",
    "priority_match",
    "oldest_compatible_match",
]

base_tags = ["new_type_env", "simopt", "fit_rep"]


def run_scenario_analysis(additional_args: List[str], tags: List[str]) -> None:
    for issuing_policy, ip_tag in zip(
        heuristic_issuing_policy_targets, heuristic_issuing_policy_tags
    ):
        for a in additional_args:
            args = (
                [
                    f"policies.issuing._target_={issuing_policy}",
                    "+experiment=simple_two_product_perishable/base_case_for_input_comparisons",
                ]
                + a
                + [f"wandb.init.tags={tags+[ip_tag]}"]
            )
            command = ["python", "run_simopt.py"] + args
            subprocess.call(command)


# Lead time
vals = [0, 1, 2]
tags = base_tags + ["change_lead_time"]
additional_args = [[f"environment.env_kwargs.lead_time={x}"] for x in vals]
run_scenario_analysis(additional_args, tags)

# Maximum useful life
vals = [2, 3, 4, 5]
tags = base_tags + ["change_max_useful_life"]
age_on_arrival_distribution_probs = [[1] + [0] * (i - 1) for i in vals]
additional_args = [
    [
        f"environment.env_kwargs.max_useful_life={x}",
        f"environment.env_params.age_on_arrival_distribution_probs={y}",
    ]
    for x, y in zip(vals, age_on_arrival_distribution_probs)
]
run_scenario_analysis(additional_args, tags)

# Product probabilities
vals = [[0.1, 0.9], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.9, 0.1]]
tags = base_tags + ["change_product_probabilities"]
additional_args = [[f"environment.env_params.product_probabilities={x}"] for x in vals]
run_scenario_analysis(additional_args, tags)

# shortage costs
vals = [7, 14, 21, 28, 35]
tags = base_tags + ["change_shortage_costs"]
additional_args = [[f"environment.env_params.shortage_costs={[x]*2}"] for x in vals]
run_scenario_analysis(additional_args, tags)

# wastage costs
vals = [5, 10, 15, 20, 25]
tags = base_tags + ["change_wastage_costs"]
additional_args = [[f"environment.env_params.wastage_costs={[x]*2}"] for x in vals]
run_scenario_analysis(additional_args, tags)

# Mean daily demand
vals = [4, 8, 16, 32]
max_order_quantity = [10, 17, 28, 49]
tags = base_tags + ["change_mean_daily_demand"]
additional_args = [
    [
        f"environment.env_params.mean_daily_demand={x}",
        f"environment.env_kwargs.max_order_quantity={y}",
    ]
    for x, y in zip(vals, max_order_quantity)
]
run_scenario_analysis(additional_args, tags)

# Substitution cost
vals = [0.5, 1, 2, 4]
tags = base_tags + ["change_substitution_cost"]
additional_args = [[f"environment.env_params.max_substitution_cost={x}"] for x in vals]
run_scenario_analysis(additional_args, tags)
