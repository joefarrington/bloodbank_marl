import subprocess

eval_args = [
    "evaluation.seed=10191",
    "pymoo.num_test_rollouts=10000",
    "evaluation.record_overall_metrics_per_eval_rollout=True",
    "wandb.init.project=de_moor_pymoo_evaluation",
]

basic_command = [
    "python",
    "/home/joefarrington/CDT/bloodbank_marl/bloodbank_marl/pymoo/multi_agent/run_pymoo.py",
]

# m=2, L=1
# best run_id = gof0rkpv
args = [
    "pymoo.algorithm.crossover.eta=2.703876643301661",
    "pymoo.algorithm.crossover.prob=0.9120554679475432",
    "pymoo.algorithm.mutation.sigma=0.03142191920494161",
    "pymoo.problem.xu=0.0254680086333281",
    "+experiment=de_moor_perishable/m2/exp2/nsgaii_fit_both",
]
args = args + eval_args
subprocess.call(basic_command + args)

# m=2, L=2
# best run_id = g4hlkfw5
args = [
    "pymoo.algorithm.crossover.eta=1.2584237752694434",
    "pymoo.algorithm.crossover.prob=0.316139736464111",
    "pymoo.algorithm.mutation.sigma=0.1278544800008214",
    "pymoo.problem.xu=0.22873235988584265 ",
    "+experiment=de_moor_perishable/m2/exp6/nsgaii_fit_both",
]
args = args + eval_args
subprocess.call(basic_command + args)

# m=5, L=1
# best run_id = u295por2
args = [
    "pymoo.algorithm.crossover.eta=5.655180759140111 ",
    "pymoo.algorithm.crossover.prob=0.9765376699178122",
    "pymoo.algorithm.mutation.sigma=0.4789623467706654",
    "pymoo.problem.xu=0.6394452080807933",
    "+experiment=de_moor_perishable/m5/exp2/nsgaii_fit_both",
]
args = args + eval_args
subprocess.call(basic_command + args)

# m=5, L=2
# best run_id = 54zpwbzg
args = [
    "pymoo.algorithm.crossover.eta=25.132766451848543",
    "pymoo.algorithm.crossover.prob=0.29227397079952533",
    "pymoo.algorithm.mutation.sigma=0.17130810212777997",
    "pymoo.problem.xu=0.030461018956530195",
    "+experiment=de_moor_perishable/m5/exp6/nsgaii_fit_both",
]
args = args + eval_args
subprocess.call(basic_command + args)
