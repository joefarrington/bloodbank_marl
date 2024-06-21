import subprocess
from typing import List

scenario_settings = ["m2/exp2", "m2/exp6", "m5/exp2", "m5/exp6"]

policies_fit = [
    "fit_replenishment_order_up_to",
    "fit_issue",
    "fit_both_replenishment_order_up_to",
]
fit_method = ["simple_ga", "open_es", "ppo"]
benchmark = ["value_iteration", "simopt"]

# Benchmarks are replenishment only
args_per_exp = [
    [f"+experiment=single_product_perishable/{b}/{s}/rep_policy.yaml"]
    for s in scenario_settings
    for b in benchmark
]
for args in args_per_exp:
    command = ["python", "run_eval.py"] + args
    subprocess.call(command)

# Fitting NN policies
args_per_exp = [
    [f"+experiment=single_product_perishable/{p}/{s}/{f}.yaml"]
    for p in policies_fit
    for s in scenario_settings
    for f in fit_method
]

for args in args_per_exp:
    command = ["python", "run_eval.py"] + args
    subprocess.call(command)
