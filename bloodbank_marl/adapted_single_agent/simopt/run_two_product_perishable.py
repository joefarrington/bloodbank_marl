import subprocess

issuing_policies = ["exact_match", "priority_match", "oldest_compatible_match"]

for i in issuing_policies:
    args = [
        f"+experiment=two_product_perishable/base_case_{i}",
    ]
    command = ["python", "run_simopt.py"] + args
    subprocess.call(command)
