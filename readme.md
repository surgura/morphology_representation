# Modular robot vector representation

## Reproducing
The following steps (approximately) reproduce the data used in the paper.

In all scripts the `parallelism` parameter is proportionate to the number of CPU cores used.

### 0. Set up the project
Clone the project:

```shell
git clone https://github.com/surgura/morphology_representation.git
```

Navigate into the cloned directory.
We recommend creating a virtual environment, e.g. using [venv](https://docs.python.org/3/library/venv.html).
\
Install the required packages:

```shell
pip install -r ./requirements.txt
```

The project has now been set up.
\
See [config.py](config.py) for all settings used in this project.

### 1. Find novel trees
Create the set of novel trees using novelty search, which will be used for training the representation network:

```shell
python find_novel_trees.py --runs all
```

### 2. Train representations
Train all representation networks using the created set of novel trees:

```shell
python train_representation.py --runs all --t_dims all --v_dims all --parallelism <choose integer>
```

### 3. Measure locality of representations
Measure the locality property of each representation:

```shell
python measure_locality.py --runs all --t_dims all --v_dims all --parallelism <choose integer>
```

### 4. Plot measured localities
Make a 3D plot of locality, t_dim, and r_dim. The script creates a seperate plot for each run, as well as an averaged plot.

```shell
python plot_locality.py
```

### 5. Select best and worst representation
Inspect the measured locality properties and select the best and worst.

```shell
python select_representations.py
```

### 6. Optimize robots using rtgae representation
Run an evolutionary algorithm, optimizing modular robots for displacement, using a benchmark setup used by the Computationally Intelligence Group at Vrije Universiteit. In short, the controller is a central pattern generator(an open loop signal generator), the body and brain genotype are both a compositional pattern-producing network.

```shell
python opt_robot_displacement_benchmark.py --runs all --parallelism <choose integer>
```

### 7. Optimize robots using CPPN representation
Run the same evolutionary algorithm, but using the selected representations as body genotypes.

```shell
python opt_robot_displacement_rtgae.py --runs all --parallelism <choose integer>
```

### TODO compare?