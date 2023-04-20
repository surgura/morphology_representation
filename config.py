from os import path

RESULTS_BASE = "results"
RUNS = 1

# settings for find_novel_trees.py
FNT_RNG_SEED = 120834789
FNT_INITIAL_MUTATIONS = 20
FNT_MUTATE_N = 5
FNT_MUTATE_P = 0.5
FNT_POPULATION_SIZE = 200
FNT_OFFSPRING_SIZE = 200
FNT_NUM_GENERATIONS = 100
FNT_OUT = lambda run: path.join(RESULTS_BASE, f"run{run}", "novel_trees")
FNT_BEST = lambda run: path.join(FNT_OUT(run), f"{FNT_NUM_GENERATIONS}.pickle")

# settings for train_representation.py
TRAIN_RNG_SEED = 23875987872
TRAIN_OUT = lambda run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"representation/t_dim{t_dim}___r_dim{r_dim}/model.state",
)
TRAIN_EPOCHS = 3000

# settings for representation model
MODEL_T_DIMS = [1, 2, 4, 8, 16, 32, 64]  # tree encoding dimensionality. 'dim' in rtgae.
MODEL_R_DIMS = [
    1,
    2,
    4,
    8,
    16,
    32,
    64,
]  # representation dimensionality. 'dim_vae' in rtgae.
MODEL_MAX_MODULES = 10
MODEL_REPR_DOMAIN = [-1.0, 1.0]

# settings for find_dd_eval_set_cutoff.py
DDEVSETCUT_SEED = 3495119999991
DDEVSETCUT_NUM_SAMPLES = 10000000
DDEVSETCUT_OUT_BASE = lambda run: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"distance_distortion_eval_set_cutoff",
)
DDEVSETCUT_OUT_BASE_R_DIM = lambda run, r_dim: path.join(
    DDEVSETCUT_OUT_BASE(run), f"r_dim{r_dim}"
)
DDEVSETCUT_OUT_CUTOFF = lambda run, r_dim: path.join(
    DDEVSETCUT_OUT_BASE_R_DIM(run, r_dim), "cutoff.pickle"
)
DDEVSETCUT_OUT_CDFPLOT_INDIVIDUAL = lambda run, r_dim: path.join(
    DDEVSETCUT_OUT_BASE_R_DIM(run, r_dim), "cdf.svg"
)
DDEVSETCUT_OUT_CDFPLOT_TOGETHER = lambda run: path.join(
    DDEVSETCUT_OUT_BASE(run), "cdf.svg"
)
DDEVSETCUT_CUTOFF = 0.99

# settings for make_dd_eval_set.py
DDEVSET_NUM_PAIRS = 10000
DDEVSET_OUT = lambda run, r_dim: path.join(
    RESULTS_BASE, f"run{run}", f"distance_distortion_eval_set/r_dim{r_dim}/set.pickle"
)
DDEVSET_MAX_FAILS = 1000
DDEVSET_SEED = 230920001435

# setting for plot_dd_eval_set_pdf.py
DDEVSETPLOT_NUM_BINS = 20

# settings for measure_distance_distortion.py
MDD_OUT = lambda run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"distance_distortion/t_dim{t_dim}___r_dim{r_dim}/measure.pickle",
)

# settings for select_representations.py
SREP_OUT = lambda run: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"selected_reps/selection.pickle",
)

# settings for plot_distance_distortion.py
PDD_OUT_COMBINED_RUNS = path.join(
    RESULTS_BASE,
    f"distance_distortion_plot/distance_distortion.svg",
)

PDD_OUT_INDIVIDUAL_RUNS = lambda run: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"distance_distortion_plot/distance_distortion.svg",
)

# settings for opt_robot_displacement_*.py
ROBOPT_RUNS = 5

ROBOPT_NUM_INITIAL_MUTATIONS = 500

ROBOPT_POPULATION_SIZE = 100
ROBOPT_OFFSPRING_SIZE = 100
ROBOPT_NUM_GENERATIONS = 100

ROBOPT_SIMULATION_TIME = 30
ROBOPT_SAMPLING_FREQUENCY = 5
ROBOPT_CONTROL_FREQUENCY = 60

# settings for opt_robot_displacement_benchmark.py
OPTBENCH_RNG_SEED = 9376349871923
OPTBENCH_OUT = lambda run, optrun: path.join(
    RESULTS_BASE, f"run{run}", f"opt_bench", f"optrun{optrun}"
)

# settings for opt_robot_displacement_rtgae.py
OPTRTGAE_RNG_SEED = 986576245246
OPTRTGAE_OUT = lambda run, optrun, bestorworst: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"opt_rtgae",
    "best" if bestorworst else "worst",
    f"optrun{optrun}",
)
OPTRTGAE_MUTATE_SIGMA = 0.1

# settings for plot_robots_fitness.py
PLOPT_OUT_INDIVIDUAL_OPTRUNS_BENCH = lambda run, optrun: path.join(
    RESULTS_BASE, f"run{run}", f"opt_fitness_plot/bench_optrun{optrun}.svg"
)
PLOPT_OUT_INDIVIDUAL_OPTRUNS_RTGAE = lambda run, optrun, bestorworst: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"opt_fitness_plot/rtgae_{'best' if bestorworst else 'worst'}_optrun{optrun}.svg",
)
PLOPT_OUT_MEAN_OPTRUNS_BENCH = lambda run: path.join(
    RESULTS_BASE, f"run{run}", f"opt_fitness_plot/bench_mean.svg"
)
PLOPT_OUT_MEAN_OPTRUNS_RTGAE = lambda run, bestorworst: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"opt_fitness_plot/rtgae_{'best' if bestorworst else 'worst'}_mean.svg",
)
