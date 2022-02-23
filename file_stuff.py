 
# for time stamping saved files and models 
from datetime import datetime
#import pickle
#import matplotlib.pyplot as plt
#from tensorflow.keras import utils

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
  ###TODO: spezifiziere Pfad-Namen je InputPair
  path = ""
  for el in sorted(options.keys()):
    if not (el=='CL' and options[el].getV() in ['n','r']):
      path += options[el].getString()
      path += "/"
  
  now = datetime.now() # current date and time
  date_time = now.strftime("%m:%d:%Y_%H:%M:%S")
  relative_file_path = DIR_PATH + path + date_time
  mkdirP(relative_file_path)
  
  #if (len(options)>8):
  #  ii=0
  #  for el in options:
  #    print("Argument {}: {}".format(ii,el))
  #    ii+=1
  #  print("You might have changed the amount of allowed arguments and forgot to update file_stuff.py")
  ##determine a unique and illustrative file path and name
  #AR="NeuralNetworkArchitecture_"+str(options['AR'])+"/"
  #GT=''
  #if (options['GT']=='i'):
  #  GT="DataInterpretation_integer/"
  #else:
  #  GT="DataInterpretation_probabilistic/"
  #CL=''
  #if (options['CL']=='d'):
  #  CL="ClassDefinition_direct/"
  #else:
  #  CL="ClassDefinition_granular_"+str(options['B'])+"_bricks/"
  #S="NumberOfSections_"+str(options['S'])+"/"
  #EP="NumberOfEpochs_"+str(options['EP'])+"/"
  #TS="NumberOfTimeSteps_"+str(options['TS'])+"/"
  #DT="BasedOnFile_"+options['DT']+"/"
    
  #now = datetime.now() # current date and time
  #date_time = now.strftime("%m:%d:%Y_%H:%M:%S")
  #relative_file_path = DIR_PATH+AR+GT+CL+S+EP+TS+DT+date_time
  #mkdirP(relative_file_path)
  
#  ###TODO: das hier muss in training stuff
#  raw_data = pickle.load(open(options['DT'], "rb"))
#  
#  def myGet_dim(list):
#    dim=[]
#    while type(list) == type([]):
#      dim.append(len(list))
#      list=list[0]
#    return dim
#  
#  if type(raw_data) != dict:
#    # set up standardized data format as dictionary
#    # this concerns all scenario data files older than 18th Nov. '21.
#    data_as_dict = {}
#    data_as_dict['scene'] = raw_data
#    data_as_dict['vmin'] = 1.0 # default this. 
#    data_as_dict['vmax'] = 10.0 ###TODO: default this. however, when committing, make sure, al data_creators behave as expected!
#    data_as_dict['dimension'] = myGet_dim(raw_data)
#    raw_data = data_as_dict
  
  return relative_file_path
