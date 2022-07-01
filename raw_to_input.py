#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:11:25 2021

@author: kraemer


This module is capable of working with raw scenario data files.
It takes them and extracts single pictures out of it which will be the input stream
for any kind of neural network. It further extracts the raw meta information to
provide it to the next module (either classsification or regression) which will
then convert this raw information into the output stream of the neural network.
"""
import numpy as np

### TODO: es wäre sinnvoll, einmal generierte Daten zu speichern und falls sie bereits existieren, diese zu laden und zu nuzen anstatt die selben Daten nochmal zu generieren. Steuerung des nötigen Verhaltens eventuell auch von höherer Stelle aus.

def extractFrames(pic_size,scenario,isIntegerInterpretation,method=0):
  # this method returns
  #   * pictures (list of numpy arrays) and
  #   * the number of present waves for [each wave speed = key] for [each picture = index] (list of dictionaries).
  #
  # here, we need the kind of data interpretation,   GT = 'i' or 'p'
  #
  # 'height' simply defines how many time steps yield a picture.
  # the width of a picture correlates directly to the scenario
  # 'method' decides what strategy we use to set the "ground truth"
  # method==0 means, we take the most actual dictionary as a truth.
  # vermutlich werden wir keine andere Methode implementieren. es gibt genug anderes zu untersuchen..
  (height,width) = pic_size
  frame_list = np.ndarray([scenario.meta_.size_ - (height-1),height,width,1])
  raw_label_list = []
  if method == 0:
    print(end='')
  else:
    print("In extract_frames(): method not implemented yet.")
    method=0
  for stop in range(height,scenario.meta_.size_+1):
    first = stop - height
    last = stop - 1
    #print(str(first)+" "+str(last))
    pic = scenario.data_[first:stop]
    reshaped_pic = pic.reshape(1,pic.shape[0],pic.shape[1],1)
    frame_list[first] = reshaped_pic
    if (method == 0):
      if isIntegerInterpretation:
        raw_label_list.append(scenario.meta_.iData_[last])
      else:
        raw_label_list.append(scenario.meta_.pData_[last])
  return frame_list,raw_label_list