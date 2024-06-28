import subprocess

exps = [
    "eighth_demand/issue_oldest_compatible_match_max_sub_wastage_alt_rula",
    "eighth_demand/issue_priority_match_max_sub_shortage_alt_rula",
    "full_demand/issue_priority_match_max_sub_wastage_alt_rula",
    "full_demand/issue_priority_match_max_sub_shortage_alt_rula",
]

for e in exps:
    command = ["python", "run_eval.py"] + [f"experiment={e}"]
    subprocess.call(command)
