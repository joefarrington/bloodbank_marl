import subprocess

demand_settings = ["full", "eighth"]
max_subtitution_costs = ["shortage", "wastage"]
issuing_policies = ["exact_match", "priority_match", "oldest_compatible_match"]

args_per_exp = [
    [f"+experiment=eight_product_perishable/{d}_demand/{i}_max_sub_{s}_alt_rula"]
    for d in demand_settings
    for s in max_subtitution_costs
    for i in issuing_policies
]

for args in args_per_exp:
    command = ["python", "run_simopt.py"] + args
    subprocess.call(command)
