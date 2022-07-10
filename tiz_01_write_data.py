# Write the training and validation data so separate files.


import argument_stuff as USER
import file_stuff as MMRY
import training_stuff as NN
import pickle
from time import time
import os



# Some code from Aaron
t0 = time()
training_options = USER.getRegressionArgumentList()
USER.process(training_options)
output_path = MMRY.preProcess(training_options, 'REG_TEST/')
network = NN.NeuralNetwork(training_options)
network.getTrainingData()
t1 = time()

print('Whatever finished. Elapsed time = {:.3f}s. Now pickle dumb the data.'.format(t1 - t0))


## Pickle dumb the data
basepath = os.getcwd()

# Set and possibl create datapath
path_to_file = basepath + '/' + 'DATA/tiz/'
os.makedirs(path_to_file, exist_ok=True)

# Pickle dump it
pickle.dump(network.trnData[0], open(path_to_file + 'X_tr.pickle', "wb"))
pickle.dump(network.trnData[1], open(path_to_file + 'y_tr.pickle', "wb"))
pickle.dump(network.valData[0], open(path_to_file + 'X_val.pickle', "wb"))
pickle.dump(network.valData[1], open(path_to_file + 'y_val.pickle', "wb"))








