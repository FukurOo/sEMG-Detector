#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 11:31:12 2021

@author: kraemer


This module provides the interface between raw sEMG wave measurements and 
REGRESSION Neural Networks. I.e. it relates the data to input and outputstreams
of the NN.
Keras-functionalities that relate (or are adjusted) specifically for REGRESSION
Neural Networks can be found here.

This module might also save/set some settings for later use in other modules.
"""

import numpy as np 
import basics
#import sys

def get_output(list_of_dicts,is_integer_interpretation,n_intervals,velocities,waves_counted_probabilistic,maxW=0):
  '''
  In this method, we sort all velocities in list_of_dicts to their respective interval.
  ##There are different encodings, according to the users preferences:
   ## integer
   ##   -> one-hot encoding
   ## probabilistic  NOT IMPEMENTED AND WILL NOT! this encoding might be used in REGRESSION TASK, still to be implemented completely.
    ##  -> prob. distrib.
 
  Main goal of this function is to create the output stream (truth of a NN with which it is trained)
  
  For example: dict1 = {1.0: 3, 4.0: 1, 7.0: 0, 10.0: 0} -- should relate to truth -->  (4/mxmm,0) (bei n_intervals=2)
  
    y  A
    |  ö                      This figure shall symbolize that there is a 
    |  X                      continuos range of values in y-direction. I.e.,
    | -X           oö         y(x) can be any float s \in [0,1]. However, x is
    | XX_-ö-______-XX_        discretized. It is one of finitely many equidist.
    |-,-------,------,-> x    values in {v_min + i*dx, i = 0,..,N}. Hier N=15
      1      5.5    10  [m/s] und v_min=1, v_max=10.
  
  
  '''
  
  '''
  Achtung: es gibt zwei verschiedene integer-Iterpretationen:
    * einmal das Zählen der maxWaves, also die Fachmann-Bewertung der Daten.
      Dieser Punkt wirkt sich dahingehend aus, dass die Maximalwellenzahl
      anders berechnet wird.   -> basics.maxwaves(list_of_dicts,integer_interpretation = True)
    * zum anderen betrachten wir
         Wahrscheinlichkeitsverteilungen je Anzahl an wellen je Geschwindigkeit
      anstatt
         einer festen Anzahl an Wellen je Geschwindigkeit
  '''
  # get the maximum number of waves that occurs in this list of dicts.
  if maxW == 0:
     maxW = basics.maxwaves(list_of_dicts,waves_counted_probabilistic)
  # für regression wollen wir als default Verhalten, dass die Anzahl an intervallen, der Anzahl an verschiedenen Geschwindigkeiten entspricht.
  # nur wenn die Anzahl an Intervallen (Uminterpretation von "S", ursprünglich Sections) explizit angegeben wird, dann nehmen wir das.
  # wir müssen dabei aber darauf acht geben, dass keines der Intervalle leer sein darf! Deshalb fordern wir, dass (n_intervals <= #velocities)! Das ergibt Sinn, da wir von einer äquidistanten generischen Geschwindigkeitsverteilung ausgehen können.
  # statt nVelocities können wir auch die komplette Liste der Geschwindigkeiten heranziehen um sie als Grundlage dieses Schrittes zu nehmen.
  if n_intervals == 0:
    # default Verhalten. (Es wurde kein Wert übergeben. Wir setzen n_intervals =#velocities)
    print("number of intervals determined from number of distinct velocities: {}".format(len(velocities)))
    n_intervals == len(velocities)
  
  length = len(list_of_dicts)
  '''
  unsure, whether this makes sense. length was undefied... copied this line from classification_task.
  ...
  tatsächlich ist diese Größe vermutlich falsch! Wir benutzen ja immer mehrere Messungen über die Zeit, um ein Bild zu definieren!
  ...
  NEIN sie ist richtig! Die Bilder wurden vorher schon generiert und die 'Eine Wahrheit' (letzter jeweiliger Zeitschritt)
      wurde bereits extrahiert und letztlich in groundtruths gepackt. Hier kommen NUR DIESE an. Das heißt, wir müssen für
      jede dieser einen Wahrheiten einen Output erstellen, und zwar GENAU EINEN.
  length = len(list_of_dicts) ist richtig.
  '''

  output = np.zeros([length,n_intervals]) # output[i] is in R^{NoI} probability distribution? no. activation distribution
  for i,el in enumerate(list_of_dicts):
    # convert dict to array, stateing the number of waves per wave speed section
    WavesPerSection = np.array(basics.dictToArray(el,n_intervals, is_integer_interpretation,velocities))
    #print(WavesPerSection)
    #if i > 7777:
    #  sys.exit()
    # unlike classification tasks, WavesPerSection is directly related to (or just is) the output.
    # However, we need to normalize all entries of it to the Interval [0,1].
    output[i] =  WavesPerSection/maxW 
  
  return maxW,output


def getNumberOfClasses(numberOfBricks,numberOfSections):
  '''
  only exists since there is such a fuction for classification tasks
  '''
  print("Warning: the first argument of regression_task.getNumberOfClasses(arg1,arg2) may have a different meaning then you might think!")
  return numberOfSections
    
def getClasses(numberOfBricks,numberOfSections):
  '''
  only exists since there is such a fuction for classification tasks
  
  however there is returned a dict like {(0,0,0): 0, ... (0,0,9):N}
  with N = getNumberOfClasses(bB,nS) entries.
  '''
  print("Warning: the first argument of regression_task.getClasses(arg1,arg2) may have a different meaning then you might think!")
  return [i for i in range(numberOfSections)]

def repairBricks(NNClass,NBricksOld,NBricksNew):
  '''
  only exists since there is such a fuction for classification tasks
  '''
  print("Warning: repairBricks() does nothing! You might wanna think about what you are doing.")
  
def investigate_new(model,network_info,step_size = 1000,showpic=False):
  #import matplotlib.pyplot as plt
  #network_info['velocities']
  sections = network_info['outputDimensionSize']
  b = np.linspace(network_info['vmin'], network_info['vmax'],sections+1)
  section_names = [str(b[i]) + " to "+str(b[i+1]) for i in range(sections)]
  inputDimension = network_info['inputDimension']
    
  x_validation,y_validation = network_info['valData']
  for num in range(0,x_validation.shape[0],step_size):
    print(num)
    print("valid_tensor: {}".format(x_validation[num:num+1]))
    
    processed_data = model(x_validation[num:num+1])
    processed_data = [el for el in processed_data.numpy()[0]]
    print(processed_data)
    tmp_truth = y_validation[num:num+1][0]
    tmp_picture = transpose(x_validation[num:num+1])
    #print("pic: \n{}".format(tmp_picture))
    if showpic:
      compare(section_names,tmp_picture,processed_data,tmp_truth)
    else:
      plot(section_names,processed_data,tmp_truth)

def plot(section_names,prediction,truth=[]):
  import matplotlib.pyplot as plt
  
  ind = np.arange(len(section_names))
  mwidth = 0.7
  if len(truth)>0:
    mwidth = mwidth/2
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.set_xlabel('Velocity range [m/s]')
  ax.set_ylabel('Muscle Activity')
  if len(truth)>0:
    ax.bar(ind-mwidth/2, prediction, color = 'b', width = mwidth)
    ax.bar(ind+mwidth/2, truth, color = 'green', width = mwidth)
    ax.legend(labels=['Model Prediction', 'Truth'])
  else: 
    ax.bar(ind, prediction, color = 'grey', width = mwidth)
    ax.legend(labels=['Model Prediction'])
  
  ax.set_xticks(ind)
  ax.set_xticklabels(section_names)
  ax.set_yticks(np.arange(0, 1, 0.1))
  plt.show()
  
def compare(section_names,orig,prediction,truth=[]):
  import matplotlib.pyplot as plt
    
  ind = np.arange(len(section_names))
  mwidth = 0.7
  if len(truth)>0:
    mwidth = mwidth/2
  fig, axs = plt.subplots(2)
  #axs[0] = fig.add_axes([0,0,1,1])
  axs[0].set_xlabel('Velocity range [m/s]')
  axs[0].set_ylabel('Muscle Activity')
  
  if len(truth)>0:
    axs[0].bar(ind-mwidth/2, prediction, color = 'b', width = mwidth)
    axs[0].bar(ind+mwidth/2, truth, color = 'green', width = mwidth)
    axs[0].legend(labels=['Model Prediction', 'Truth'])
  else: 
    axs[0].bar(ind, prediction, color = 'grey', width = mwidth)
    axs[0].legend(labels=['Model Prediction'])
  
  axs[0].set_xticks(ind)
  axs[0].set_xticklabels(section_names)
  axs[0].set_yticks(np.arange(0, 1, 0.1))
  #print(orig)
  axs[1].imshow(orig)
  
  plt.show()
  
def transpose(tensor):
  assert tensor.shape[0] == 1 and tensor.shape[-1] ==1, "data doesnt fulfill assumption."
  ret = np.ndarray((tensor.shape[1],tensor.shape[2]))
  return ret
  """ what the ?? ret ist schon fertig initialisiert! numpy scheint hier zu ahnen, was ich tun will!
  """
  #
  #for i in range(tensor.shape[1]):
  #  for j in range(tensor.shape[2]):
  #    ret[i][j] = float(tensor[0][i][j][0])
  #return ret
    