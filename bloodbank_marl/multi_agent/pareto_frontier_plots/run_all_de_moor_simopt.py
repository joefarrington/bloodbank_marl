import subprocess

experiment_configs = ["m2/exp2", "m2/exp6", "m5/exp2", "m5/exp6"]

for experiment_config in experiment_configs:
    command = [
        "python",
        "run_simopt.py",
        f"+experiment=de_moor_perishable/{experiment_config}/opt_for_pareto",
    ]
    subprocess.call(command)
