# Load the training and validation data which we pickled before

import os
import pickle

import matplotlib.pyplot as plt

# Load the data
basepath = os.getcwd()
path_to_file = basepath + '/' + 'DATA/tiz/'

with open(path_to_file + 'X_tr.pickle', "rb") as f:
  X_tr = pickle.load(f)

with open(path_to_file + 'y_tr.pickle', "rb") as f:
  y_tr = pickle.load(f)

with open(path_to_file + 'X_val.pickle', "rb") as f:
  X_val = pickle.load(f)

with open(path_to_file + 'y_val.pickle', "rb") as f:
  y_val = pickle.load(f)



# Visualize some of the training data
for idx in [0, 8, 3908, 42091]:
    plt.figure(idx)
    plt.imshow(X_tr[idx, :, :, 0].transpose(), cmap='hot', interpolation='nearest')
    plt.show()

