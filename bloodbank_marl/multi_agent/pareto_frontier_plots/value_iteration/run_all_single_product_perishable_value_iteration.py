import subprocess

experiment_configs = ["m2/exp2", "m2/exp6", "m5/exp2", "m5/exp6"]

for experiment_config in experiment_configs:
    command = [
        "python",
        "run_value_iteration.py",
        f"+experiment=single_product_perishable/{experiment_config}/pareto_vi",
    ]
    subprocess.call(command)
