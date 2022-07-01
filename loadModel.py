#import file_stuff as MMRY
import training_stuff as NN
from tensorflow import keras
import sys
import pickle
import os.path as pth


'''
 this program loads already trained NN-models in order to investigate them

 it also loads 
   the validation data
   the original wave scenarios, that are correlated to the validation objects
'''
if len(sys.argv)<2:
  print("Please specify the path to the model as far as 'datetime'!")
  sys.exit(1)
  
path = sys.argv[1]
path_to_model = path+"/model"
# Scene Map wird nur für Klassifizierung benötigt. Die Regression-Model-Ergebnisse können ohne mapping direkt mit den Validierungsdaten verglichen werden.
filename_network = path+"/network_info.pickle"
if not 'REG_TEST' in path:
#if pth.exists(filename_network):
  import classification_task as task
#  scene,valData= pickle.load(open(filename_sceneMap, "rb"))
else:
#  scene = None
  import regression_task as task
#  valData= pickle.load(open(filename_sceneMap, "rb"))
network_dict = pickle.load(open(filename_network, "rb"))
model = keras.models.load_model(path_to_model)

#print(network_dict)

task.investigate_new(model,network_dict,1000,True)