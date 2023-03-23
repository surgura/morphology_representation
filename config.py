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

# settings for measure_locality.py
MLOC_RNG_SEED = 2389471248137
MLOC_NUM_SAMPLES = 10000
MLOC_OUT = lambda run, t_dim, r_dim: path.join(
    RESULTS_BASE, f"run{run}", f"locality/t_dim{t_dim}___r_dim{r_dim}/measure.pickle"
)

# settings for select_representations.py
SREP_OUT = lambda run: path.join(
    RESULTS_BASE,
    f"run{run}",
    f"selected_reps/selection.pickle",
)
