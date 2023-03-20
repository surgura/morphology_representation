from os import path

RESULTS_BASE = "results"

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
TRAIN_OUT = lambda run: path.join(
    RESULTS_BASE, f"run{run}", "representation/model.state"
)
TRAIN_EPOCHS = 3000

# settings for representation model
MODEL_DIM = [1, 2, 4, 8, 16, 32, 64]
MODEL_DIM_VAE = [1, 2, 4, 8, 16, 32, 64]
MODEL_MAX_MODULES = 10

# settings for measure_quality_representation.py
MREP_RNG_SEED = 2389471248137
MREP_NUM_SAMPLES = 10000
