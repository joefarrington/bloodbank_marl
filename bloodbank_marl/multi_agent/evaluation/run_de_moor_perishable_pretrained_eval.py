import subprocess
from typing import List

action_output = ["order_up_to", "direct_action"]
scenario_settings = ["m2/exp6", "m5/exp6"]
fit_method = ["simple_ga", "simopt"]

args_per_exp = [
    ["+experiment=de_moor_perishable/fit_both_from_pretrained_{a}/{s}/{f}.yaml"]
    for a in action_output
    for s in scenario_settings
    for f in fit_method
]

for args in args_per_exp:
    command = ["python", "run_eval.py"] + args
    subprocess.call(command)
