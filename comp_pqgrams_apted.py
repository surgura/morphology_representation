from train_set import TrainSet
import numpy as np
from apted_util import apted_tree_edit_distance
import time

trainset = TrainSet(run=0, experiment_name="test")

n = 20000
idxs1 = np.random.randint(0, len(trainset), n)
idxs2 = np.random.randint(0, len(trainset), n)

print("apted")
start = time.time()
for i1, i2 in zip(idxs1, idxs2):
    apted_tree_edit_distance(trainset[i1][3], trainset[i2][3])
end = time.time()
aptedtime = end - start

print("pqgrams")
start = time.time()
for i1, i2 in zip(idxs1, idxs2):
    trainset[i1][2].edit_distance(trainset[i2][2])
end = time.time()
pqgramstime = end - start

print(f"{aptedtime=}, {pqgramstime=} {aptedtime/pqgramstime=}")
