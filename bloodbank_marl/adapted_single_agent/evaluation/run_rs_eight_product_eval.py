import subprocess
from typing import List

demand_settings = ["eighth_demand", "full_demand"]

# Simple GA
exps = [
    # Max sub shortage
    "fit_rep/simple_ga_max_sub_shortage_alt_rula.yaml",
    "fit_rep/simple_ga_rep_pretrained_max_sub_shortage_alt_rula.yaml",
    "fit_both/simple_ga_max_sub_shortage_alt_rula.yaml",
    "fit_both/simple_ga_rep_pretrained_max_sub_shortage_alt_rula.yaml",
    "fit_both/simple_ga_both_pretrained_max_sub_shortage_alt_rula.yaml",
    "fit_both/simple_ga_issue_pretrained_max_sub_shortage_alt_rula.yaml",
    "fit_issue/simple_ga_issue_pretrained_max_sub_shortage_alt_rula.yaml",
    "fit_issue/simple_ga_max_sub_shortage_alt_rula.yaml",
    # Max sub wastage'
    "fit_rep/simple_ga_max_sub_wastage_alt_rula.yaml",
    "fit_rep/simple_ga_rep_pretrained_max_sub_wastage_alt_rula.yaml",
    "fit_both/simple_ga_max_sub_wastage_alt_rula.yaml",
    "fit_both/simple_ga_rep_pretrained_max_sub_wastage_alt_rula.yaml",
    "fit_both/simple_ga_both_pretrained_max_sub_wastage_alt_rula.yaml",
    "fit_both/simple_ga_issue_pretrained_max_sub_wastage_alt_rula.yaml",
    "fit_issue/simple_ga_issue_pretrained_max_sub_wastage_alt_rula.yaml",
    "fit_issue/simple_ga_max_sub_wastage_alt_rula.yaml",
]

args_per_exp = [
    [f"+experiment=rs_eight_product_perishable/{d}/simple_ga/{e}"]
    for d in demand_settings
    for e in exps
]

for args in args_per_exp:
    command = ["python", "run_eval.py"] + args
    subprocess.call(command)

# SimOpt

exps = [
    "exact_match_max_sub_shortage_alt_rula",
    "priority_match_max_sub_shortage_alt_rula.yaml",
    "oldest_compatible_match_max_sub_shortage_alt_rula",
    "exact_match_max_sub_wastage_alt_rula",
    "priority_match_max_sub_wastage_alt_rula.yaml",
    "oldest_compatible_match_max_sub_wastage_alt_rula",
]

args_per_exp = [
    [f"+experiment=rs_eight_product_perishable/{d}/simopt/{e}"]
    for d in demand_settings
    for e in exps
]

for args in args_per_exp:
    command = ["python", "run_eval.py"] + args
    subprocess.call(command)
