# Modular robot vector representation

## Reproducing
The following steps (approximately) reproduce the data used in the paper.

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
python find_novel_trees --runs all
```

### 2. Train representation
Train all representation networks using the created set of novel trees:

```shell
python train_representation --runs all --t_dims all --v_dims all --jobs <choose integer>
```

### 3. Measure locality of representations
Measure the locality property of each representation:

```shell
python measure_locality --runs all --t_dims all --v_dims all --jobs <choose integer>
```