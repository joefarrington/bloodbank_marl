import subprocess

min_exact_match_pcs = [50, 75, 95]

for m in min_exact_match_pcs:
    command = [
        "python",
        "run_outer_loop.py",
        f"+experiment=two_product_perishable/min_exact_match_{m}",
    ]
    subprocess.call(command)
