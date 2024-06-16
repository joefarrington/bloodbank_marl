import subprocess
from typing import List

exps = [
    "simopt/exact_match_issuing",
    "simopt/oldest_compatible_match_issuing",
    "simopt/priority_match_issuing",
    "simple_ga/order_up_to/exact_match_issuing",
    "simple_ga/order_up_to/oldest_compatible_match_issuing",
    "simple_ga/order_up_to/priority_match_issuing",
    "simple_ga/order_up_to/simple_ga_issuing",
]

args_per_exp = [[f"+experiment=two_product_perishable/{e}"] for e in exps]


for args in args_per_exp:
    command = ["python", "run_eval.py"] + args
    subprocess.call(command)
