# Modular robot vector representation

## Reproducing
The following steps (approximately) reproduce the data used in the paper.

In all scripts the `parallelism` parameter is proportional to the number of CPU cores used.

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

### 3. Measure distance distortion of representations

#### 3.1 Create evaluation set of pairs of vector representations
```shell
python make_dd_eval_set.py --runs all --r_dims all --parallelism <choose integer>
```

#### 3.2 Measure distance distortion for each representation
```shell
python measure_distance_distortion.py --runs all --t_dims all --v_dims all --parallelism <choose integer>
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

### 6. Optimize robots using benchmark CPPN representation
Run an evolutionary algorithm, optimizing modular robots for displacement, using a benchmark setup used by the Computationally Intelligence Group at Vrije Universiteit. In short, the controller is a central pattern generator(an open loop signal generator), the body and brain genotype are both compositional pattern-producing networks.

```shell
python opt_robot_displacement_benchmark.py --runs all --optruns all --parallelism <choose integer>
```

### 7. Optimize robots using rtgae representation
Run the same evolutionary algorithm, but using the selected representations as body genotypes.

```shell
python opt_robot_displacement_rtgae.py --runs all --optruns all --parallelism <choose integer>
```

### 8. Plot optimization results
Plot the fitness over generations for both the benchmark and rtgae representations.

```shell
python plot_robopt_fitness.py
```

### 8.5 (optional) Visualize the best robots
See the best robots in action. The following script simulates the best robot of the final generation for each optimization process.

```shell
python simulate_robopt.py --run <run> --optrun <optimization run> bench
```

```shell
python simulate_robopt.py --run <run> --optrun <optimization run> rtgae <best|worst>
```
