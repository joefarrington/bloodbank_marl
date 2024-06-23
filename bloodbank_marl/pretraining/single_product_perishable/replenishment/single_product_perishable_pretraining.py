import subprocess
from typing import List

scenario_settings = ["m2/exp6", "m5/exp6"]
nn_type = ["direct_action", "order_up_to"]

args_per_exp = [
    [f"+experiment=single_product_perishable/{s}/fit_{n}_nn.yaml"]
    for s in scenario_settings
    for n in nn_type
]
for args in args_per_exp:
    command = ["python", "run_pretraining_replenishment_classification.py"] + args
    subprocess.call(command)
