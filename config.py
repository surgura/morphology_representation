from os import path

RESULTS_BASE = "results"
RUNS = 1

# settings for generate_training_set.py
GENTRAIN_RNG_SEED = 129812393433
GENTRAIN_OUT = lambda run, experiment_name: path.join(
    RESULTS_BASE, f"run{run}/exps/{experiment_name}/training_set/set.pickle"
)
GENTRAIN_ARCHIVE_SIZE = 10000
# ----old params----
# GENTRAIN_NUM_GENERATIONS = 200
# GENTRAIN_POPULATION_SIZE = 128
# GENTRAIN_OFFSPRING_SIZE = 64
# GENTRAIN_KNN_K = 5
# GENTRAIN_INITIAL_MUTATIONS = 20
# GENTRAIN_MUTATE_N = 5
# GENTRAIN_MUTATE_P = 0.5
# GENTRAIN_ARCHIVE_APPEND_NUM = 5

# setting for render_training_set.py
RENDERTRAIN_OUT = lambda run, experiment_name, item_i: path.join(
    RESULTS_BASE,
    f"run{run}/exps/{experiment_name}/training_set_render/{str(item_i).zfill(5)}.png",
)

# settings for representation model
MODEL_T_DIMS = [
    # 16,
    128,
    256,
    512,
    768,
]  # tree encoding dimensionality. 'dim' in rtgae.
MODEL_R_DIMS = [
    24,
    48,
    72,
    216,
    288,
]  # 8, 24  # representation dimensionality. 'dim_vae' in rtgae.
MODEL_MAX_MODULES = 10
MODEL_REPR_DOMAIN = [-1.0, 1.0]
MODEL_MAX_MODULES_INCL_EMPTY = (
    32  # this is temporary until rtgae code is updated to ignore empty
)

# settings for train_representation.py
TRAIN_RNG_SEED = 23875987872
TRAIN_OUT = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/trained_representation/t_dim{t_dim}___r_dim{r_dim}/model.state",
)
TRAIN_DD_OUT = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/trained_representation_dd/t_dim{t_dim}___r_dim{r_dim}/model.state",
)
TRAIN_EPOCHS = 200
TRAIN_BATCH_SIZE = 200
TRAIN_DD_TRIPLET_FACTOR = 1.0
TRAIN_OUT_LOSS = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/trained_representation/t_dim{t_dim}___r_dim{r_dim}/loss.pickle",
)
TRAIN_DD_OUT_LOSS = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/trained_representation_dd/t_dim{t_dim}___r_dim{r_dim}/loss.pickle",
)
TRAIN_DD_MARGIN = 0.05

# settings for plot_train_loss.py
PLTTRAIN_OUT = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/trained_plot/t_dim{t_dim}___r_dim{r_dim}.svg",
)

# settings for generate_evaluation_representation_set.py
GENEVALREPR_SEED = 34592349873289
GENEVALREPR_NUM_REPRESENTATIONS = 1000
GENEVALREPR_NUM_BINS = 20
GENEVALREPR_OUT_RTGAE = lambda run, experiment_name, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}/exps/{experiment_name}",
    f"evaluation/representation/rtgae/t_dim{t_dim}___r_dim{r_dim}/set.pickle",
)

# settings for generate_evaluation_solution_set.py
GENEVALSOL_RNG_SEED = 6534591999
GENEVALSOL_ARCHIVE_SIZE = 1000
GENEVALSOL_OUT = lambda run, experiment_name: path.join(
    RESULTS_BASE, f"run{run}/exps/{experiment_name}", "evaluation/solution/set.pickle"
)

# settings for measure_coverage_rtgae.py
CVGRTGAE_OUT = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/evaluation/coverage/t_dim{t_dim}___r_dim{r_dim}/coverage.pickle",
)

# settings for measure_stress_rtgae.py
STRESSRTGAE_OUT = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/evaluation/stress/t_dim{t_dim}___r_dim{r_dim}/stress.pickle",
)

# settings for plot_measures.py
PLTMSR_OUT_STRESS_INDIVIDUAL_RUNS = lambda experiment_name, run: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/evaluation/plots/stress.svg",
)

PLTMSR_OUT_COVERAGE_INDIVIDUAL_RUNS = lambda experiment_name, run: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/evaluation/plots/coverage.svg",
)

# PLTMSR_OUT_STRESS_COMBINED_RUNS = path.join(
#     RESULTS_BASE,
#     f"evaluation/stress.svg",
# )

# PLTMSR_OUT_COVERAGE_COMBINED_RUNS = path.join(
#     RESULTS_BASE,
#     f"evaluation/coverage.svg",
# )

PLTMSR_OUT_PAIRS = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/evaluation/plots/scatter___t_dim{t_dim}___r_dim{r_dim}.png",
)

# settings for select_representations.py
SREP_OUT = lambda run: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"selected_reps/selection.pickle",
)

# settings for opt_robot_displacement_*.py
ROBOPT_RUNS = 5

ROBOPT_NUM_INITIAL_MUTATIONS = 500

ROBOPT_POPULATION_SIZE = 100
ROBOPT_OFFSPRING_SIZE = 50
ROBOPT_NUM_GENERATIONS = 20

ROBOPT_SIMULATION_TIME = 30
ROBOPT_SAMPLING_FREQUENCY = 5
ROBOPT_CONTROL_FREQUENCY = 60

ROBOPT_BRAIN_INITIAL_STD = 0.5
ROBOPT_BRAIN_NUM_GENERATIONS = 20

# settings for opt_robot_displacement_benchmark.py
OPTBENCH_RNG_SEED = 9376349871923
OPTBENCH_OUT = lambda experiment_name, run, optrun: path.join(
    RESULTS_BASE, f"run{run}", f"exps/{experiment_name}/opt_cppn", f"optrun{optrun}"
)

# settings for opt_robot_displacement_rtgae.py
OPTRTGAE_RNG_SEED = 986576245246
OPTRTGAE_OUT = lambda experiment_name, run, optrun, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/opt_rtgae",
    f"t_dim{t_dim}___r_dim{r_dim}___optrun{optrun}",
)
OPTRTGAE_MUTATE_SIGMA = 0.1

# settings for opt_robot_displacement_cmaes.py
OPTCMAES_RNG_SEED = 23409587821
OPTCMAES_OUT = lambda experiment_name, run, optrun, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/opt_cmaes",
    f"t_dim{t_dim}___r_dim{r_dim}___optrun{optrun}",
)
OPTCMAES_NUM_EVALUATIONS = (
    ROBOPT_POPULATION_SIZE + ROBOPT_OFFSPRING_SIZE * ROBOPT_NUM_GENERATIONS
)
OPTCMAES_BODY_INITIAL_STD = 0.5

# settings for plot_robots_fitness.py
PLOPT_OUT_INDIVIDUAL_OPTRUNS_BENCH = lambda experiment_name, run, optrun: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/opt_fitness_plot/bench_optrun{optrun}.svg",
)
PLOPT_OUT_INDIVIDUAL_OPTRUNS_RTGAE = lambda experiment_name, run, optrun, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/opt_fitness_plot/rtgae_t_dim{t_dim}___r_dim{r_dim}___optrun{optrun}.svg",
)
PLOPT_OUT_INDIVIDUAL_OPTRUNS_CMAES = lambda experiment_name, run, optrun, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/opt_fitness_plot/cmaes_t_dim{t_dim}___r_dim{r_dim}___optrun{optrun}.svg",
)
PLOPT_OUT_MEAN_OPTRUNS_BENCH = lambda experiment_name, run: path.join(
    RESULTS_BASE, f"run{run}", f"exps/{experiment_name}/opt_fitness_plot/bench_mean.svg"
)
PLOPT_OUT_MEAN_OPTRUNS_RTGAE = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/opt_fitness_plot/rtgae_t_dim{t_dim}___r_dim{r_dim}___mean.svg",
)
PLOPT_OUT_MEAN_OPTRUNS_CMAES = lambda experiment_name, run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/opt_fitness_plot/cmaes_t_dim{t_dim}___r_dim{r_dim}___mean.svg",
)
PLOPT_OUT_ALL = lambda experiment_name, run: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/opt_fitness_plot/all.svg",
)

# settings for sample_representations.py
SAMPLEREPR_OUT_CENTER = lambda experiment_name, run, t_dim, r_dim, i: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/evaluation/samples/t_dim{t_dim}___r_dim{r_dim}/{i}.png",
)

# settings for plot_crossover.py
PLTXOVER_OUT = lambda experiment_name, run, t_dim, r_dim, tag: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"exps/{experiment_name}/evaluation/xover/t_dim{t_dim}___r_dim{r_dim}/xo_{tag}.png",
)
