import subprocess
from typing import List

scenario_settings = ["m2/exp6", "m5/exp6"]
action_output = ["order_up_to", "direct_action"]
fit_method = ["simple_ga", "open_es"]

args_per_exp = [
    [
        f"+experiment=single_product_perishable/fit_both_from_pretrained_replenishment_{a}/{s}/{f}.yaml"
    ]
    for s in scenario_settings
    for a in action_output
    for f in fit_method
]

for args in args_per_exp:
    command = ["python", "run_eval.py"] + args
    subprocess.call(command)
