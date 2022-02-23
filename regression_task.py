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

def get_output(list_of_dicts,is_integer_interpretation,n_intervals,velocities,maxW=0):
  '''
  In this method, we sort all velocities in list_of_dicts to their respective interval.
  ##There are different encodings, according to the users preferences:
   ## integer
   ##   -> one-hot encoding
   ## probabilistic  NOT IMPEMENTED AND WILL NOT! this encoding might be used in REGRESSION TASK, still to be implemented completely.
    ##  -> prob. distrib.
 
  Main goal of this function is to create the output stream (truth of a NN with which it is trained)
  
  For example: dict1 = {1.0: 3, 4.0: 1, 7.0: 0, 10.0: 0} -- should relate to truth -->  (4/mxmm,0) (bei n_intervals=2)
  
    y A
    |  ö                      This figure shall symbolize that there is a 
    |  X                      continuos range of values in y-direction. I.e.,
    | -X           oö         y(x) can be any float s \in [0,1]. However, x is
    | XX_-ö-______-XX_        discretized. It is one of finitely many equidist.
    |-,-------,------,-> x    values in {v_min + i*dx, i = 0,..,N}. Hier N=15
      1      5.5    10  [m/s] und v_min=1, v_max=10.
  
  
  '''
    
  # get the maximum number of waves that occurs in this list of dicts.
  if maxW == 0:
    length = len(list_of_dicts)
    count = np.ndarray([length],dtype=np.int32)
    for i in range(length):
      count[i] = 0
      for kel in list_of_dicts[i].keys():
        if is_integer_interpretation:
          count[i] += int(list_of_dicts[i][kel])
        else: # probabilistic interpretation leads to a more complex data structure here:
          count[i] += int(max(list_of_dicts[i][kel].keys()))
      if count[i] > maxW:
        maxW = int(count[i])
        if type(maxW)!=type(2):
          print("typ von maxW Zeile 60 falsch")
    #maxW = maxW
  
  
  # für regression wollen wir als default Verhalten, dass die Anzahl an intervallen, der Anzahl an verschiedenen Geschwindigkeiten entspricht.
  # nur wenn die Anzahl an Intervallen (Uminterpretation von "S", ursprünglich Sections) explizit angegeben wird, dann nehmen wir das.
  # wir müssen dabei aber darauf acht geben, dass keines der Intervalle leer sein darf! Deshalb fordern wir, dass (n_intervals <= #velocities)! Das ergibt Sinn, da wir von einer äquidistanten generischen Geschwindigkeitsverteilung ausgehen können.
  # statt nVelocities können wir auch die komplette Liste der Geschwindigkeiten heranziehen um sie als Grundlage dieses Schrittes zu nehmen.
  if n_intervals == 0:
    # default Verhalten. (Es wurde kein Wert übergeben. Wir setzen n_intervals =#velocities)
    print("number of intervals determined from number of distinct velocities: {}".format(len(velocities)))
    n_intervals == len(velocities)

  output = np.zeros([length,n_intervals]) # output[i] is in R^{NoI} probability distribution? no. activation distribution
  for i,el in enumerate(list_of_dicts):
    # convert dict to array, stateing the number of waves per wave speed section
    WavesPerSection = np.array(basics.dictToArray(el,n_intervals, is_integer_interpretation,velocities))
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