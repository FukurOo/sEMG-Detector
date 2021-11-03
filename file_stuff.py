 
# for time stamping saved files and models 
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras import utils
#import subprocess
import os

def mkdirP(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def preProcess(options,DIR_PATH='TEST/'):
  if (len(options)>8):
    ii=0
    for el in options:
      print("Argument {}: {}".format(ii,el))
      ii+=1
    print("You might have changed the amount of allowed arguments and forgot to update file_stuff.py")
  #determine a unique and illustrative file path and name
  AR="NeuralNetworkArchitecture_"+str(options['AR'])+"/"
  GT=''
  if (options['GT']=='i'):
    GT="DataInterpretation_integer/"
  else:
    GT="DataInterpretation_probabilistic/"
  CL=''
  if (options['CL']=='d'):
    CL="ClassDefinition_direct/"
  else:
    CL="ClassDefinition_granular_"+str(options['B'])+"_bricks/"
  S="NumberOfSections_"+str(options['S'])+"/"
  EP="NumberOfEpochs_"+str(options['EP'])+"/"
  TS="NumberOfTimeSteps_"+str(options['TS'])+"/"
  DT="BasedOnFile_"+options['DT']+"/"
    
  now = datetime.now() # current date and time
  date_time = now.strftime("%m:%d:%Y_%H:%M:%S")
  relative_file_path = DIR_PATH+AR+GT+CL+S+EP+TS+DT+date_time
  mkdirP(relative_file_path)
  
  raw_data = pickle.load(open(options['DT'], "rb"))
  
  if type(raw_data[0]) == list: # depending on the data creator, we saved stuff too encapsulated. done so in data_creator_001. This needs better attention in the future.
    raw_data = raw_data[0]
  
  return raw_data,relative_file_path
  
def postProcess(relative_file_path,history,model,scene_mapping):
  #print(history.history.keys())
  
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(relative_file_path+"/Accuracy")
  plt.savefig(relative_file_path+"/Accuracy.eps")
  plt.clf()
  
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(relative_file_path+"/Loss")
  plt.savefig(relative_file_path+"/Loss.eps")
  plt.clf()
  
  model.save(relative_file_path+"/model")
  utils.plot_model(model, to_file=(relative_file_path+"/model.png"))
  
  scene_file_name = relative_file_path+"/scene_mapping.pickle"
  pickle.dump(scene_mapping, open(scene_file_name, "wb"))
  
  print("\n\n\n\n\n\n\n")
  print("Finishing up. Proceede by executing '. ./execute-me.sh <outfile>'.")
  command = "mv $1 "+relative_file_path+"/."+" && echo '... proceeding execution ...' && echo 'Done.'"
  write_c_to_file = "echo '"+command+"' > execute-me.sh"
  print()
  os.system(write_c_to_file)
  os.system("sleep 0.02")
  #subprocess.run([write_c_to_file])
#  print(file_structure_info)