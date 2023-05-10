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

### 1. Create RTGAE training set
Create a set of robot trees which will be used for training the representation network:
```shell
python generate_training_set.py -r all --parallelism <choose integer>
```
#### 1.1 (optional) Render RTGAE training set
Render images of the training set:
```shell
python render_training_set.py -r all
```

### 2. Train RTGAE representations
Train all representation networks using the created set of novel trees:

```shell
python train_representation.py --runs all --t_dims all --r_dims all --parallelism <choose integer>
```

### 3. Measure representation quality
#### 3.1 Create solution evaluation set
Create a set of robot trees which will be used to measure coverage.
```shell
python generate_evaluation_solution_set.py
```
#### 3.2 Create representation evaluation set
```shell
python generate_evaluation_representation_set.py --parallelism <choose integer>
```
#### 3.3 Measure RTGAE coverage
```shell
python measure_coverage_rtgae.py --runs all --t_dims all --r_dims all --parallelism <choose integer>
```
#### 3.4 Measure CPPN coverage
```shell
python measure_coverage_cppn.py --runs all --parallelism <choose integer>
```
#### 3.5 Measure RTGAE stress TODO what measure
```shell
python measure_stress_rtgae.py --runs all --t_dims all --r_dims all --parallelism <choose integer>
```
#### 3.6 Measure RTGAE stress TODO what measure
```shell
python measure_stress_cppn.py --runs all --parallelism <choose integer>
```

### 4 Optimize robot using different represenations
### 4.1 Using benchmark CPPN representation
Run an evolutionary algorithm, optimizing modular robots for displacement, using a benchmark setup used by the Computationally Intelligence Group at Vrije Universiteit. In short, the controller is a central pattern generator(an open loop signal generator), the body and brain genotype are both compositional pattern-producing networks.

```shell
python opt_robot_displacement_cppn.py --runs all --optruns all --parallelism <choose integer>
```
### 4.2 Using RTGAE representation
Run the same evolutionary algorithm, but using the selected representations as body genotypes.

```shell
python opt_robot_displacement_rtgae.py --runs all --t_dims all --r_dims all --optruns all --parallelism <choose integer>
```

### 5. Plot optimization results
Plot the fitness over generations for both the benchmark and rtgae representations.

```shell
python plot_robopt_fitness.py
```
### 5.1 (optional) Visualize the best robots
See the best robots in action. The following script simulates the best robot of the final generation for each optimization process.

```shell
python simulate_robopt.py --run <run> --optrun <optimization run> cppn
```

```shell
python simulate_robopt.py --run <run> --optrun <optimization run> rtgae --v_dim <choose integer> --r_dim <choose integer>
```
