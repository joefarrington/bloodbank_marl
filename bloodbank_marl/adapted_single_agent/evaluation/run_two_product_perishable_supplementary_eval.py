import subprocess
from typing import List

exps = [
    "simple_ga/order_up_to/supplementary_eval/fit_both_rep_exact_match_issuing",
    "simple_ga/order_up_to/supplementary_eval/fit_both_rep_priority_match_issuing",
    "simple_ga/order_up_to/supplementary_eval/fit_both_rep_oldest_compatible_match_issuing",
    "simple_ga/order_up_to/supplementary_eval/fit_rep_exact_match_issuing_from_fit_both",
    "simple_ga/order_up_to/supplementary_eval/fit_rep_priority_match_issuing_from_fit_both",
    "simple_ga/order_up_to/supplementary_eval/fit_rep_oldest_compatible_match_issuing_from_fit_both",
]

args_per_exp = [[f"+experiment=two_product_perishable/{e}"] for e in exps]


for args in args_per_exp:
    command = ["python", "run_eval.py"] + args
    subprocess.call(command)
