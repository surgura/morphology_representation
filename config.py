from os import path

RESULTS_BASE = "results"
RUNS = 1

# settings for generate_training_set.py
GENTRAIN_RNG_SEED = 129812393433
GENTRAIN_OUT = lambda run: path.join(
    RESULTS_BASE, f"run{run}", "training_set/set.pickle"
)
GENTRAIN_NUM_GENERATIONS = 200
GENTRAIN_POPULATION_SIZE = 128
GENTRAIN_OFFSPRING_SIZE = 64
GENTRAIN_ARCHIVE_SIZE = 1024
GENTRAIN_KNN_K = 5
GENTRAIN_INITIAL_MUTATIONS = 20
GENTRAIN_MUTATE_N = 5
GENTRAIN_MUTATE_P = 0.5
GENTRAIN_ARCHIVE_APPEND_NUM = 5

# setting for render_training_set.py
RENDERTRAIN_OUT = lambda run, item_i: path.join(
    RESULTS_BASE, f"run{run}", f"training_set_render/{item_i}.png"
)

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

# settings for train_representation.py
TRAIN_RNG_SEED = 23875987872
TRAIN_OUT = lambda run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"representation/t_dim{t_dim}___r_dim{r_dim}/model.state",
)
TRAIN_EPOCHS = 3000

# settings for generate_evaluation_set.py
GENEVAL_SEED = 34592349873289
GENEVAL_NUM_REPRESENTATIONS = 100
GENEVAL_NUM_BINS = 20
GENEVAL_OUT_RTGAE = lambda run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"evaluation_set/rtgae/t_dim{t_dim}___r_dim{r_dim}/set.pickle",
)

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

PDD_OUT_SCATTER = lambda run, t_dim, r_dim: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"distance_distortion_plot/scatter/t_dim{t_dim}___r_dim{r_dim}.svg",
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
