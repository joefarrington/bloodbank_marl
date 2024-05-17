import subprocess

eval_args = [
    "evaluation.seed=10191",
    "pymoo.num_test_rollouts=10000",
    "evaluation.record_overall_metrics_per_eval_rollout=True",
    "wandb.init.project=simple_two_product_perishable_pymoo_evaluation",
]

basic_command = [
    "python",
    "run_pymoo.py",
]

# Fit all three with NSGA-II
# best run_id = x4zb34ah
args = [
    "pymoo.algorithm.crossover.eta=3.8955397208252394 ",
    "pymoo.algorithm.crossover.prob=0.6531581231575427",
    "pymoo.algorithm.mutation.sigma=0.085438441106315",
    "pymoo.problem.xu=0.35919000437414694",
]
args = args + eval_args
subprocess.call(basic_command + args)
