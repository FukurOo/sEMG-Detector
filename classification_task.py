#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 10:47:38 2021

@author: kraemer


This module provides the interface between raw sEMG wave measurements and 
CLASSIFICATION Neural Networks. I.e. it relates the data to input and output-
streams of the NN.
Keras-functionalities that relate (or are adjusted) specifically for CLASSIFICATION
Neural Networks can be found here.

basically it uses the raw meta data to map them to the NN-classes. It also saves
some settings like the mapping itself for later use in other modules.
"""
import numpy as np 
import scipy as sp
from scipy import special
import matplotlib.pyplot as plt
import basics

def getNumberOfClasses(numberOfBricks,numberOfSections):
  '''
  calculate and return the number of NN classes.

  Consider the following picture. There, the number of bricks is 7. The number
  of sections is 4. (sometimes we may use the words 'granularity' or 'brick size'
  which here is 100/7)
  Every possible state of how the graph could look like, is a class. In this
  graph, there are 5 bricks ('XXXX') out of 7 possible. This states name
  (or the name of the class) is '(0,3,2,0)'.
  
  Note: the number of bricks can be equal to the number of maximal waves.
  However, this is not necessarily the case!
  
  nOB_| ____ ____ ____ ____
      ||____|____|____|____|
      ||____|____|____|____|
      ||____|____|____|____|
      ||____|____|____|____|
      ||____|XXXX|____|____|
      ||____|XXXX|XXXX|____|
      ||____|XXXX|XXXX|____|__>
  
  We can calculate the number of classes explicitly. The number of NN classes is:   
  $$\sum_{n=0}^{N_B} \frac{(N_S + n-1)!}{n! (N_S - 1)!)} $$
  '''
  total_classes = 0
  for n in range(numberOfBricks+1):
    total_classes += sp.special.comb(numberOfSections + n - 1, n, exact=True)
    #print(sp.special.comb(numberOfSections + n - 1, n, exact=True))
  if total_classes >= 500:
    print("\n\nWarning! This Neural Network is trained with a lot of classes! ({})\n\n".format(total_classes))
  return total_classes
    
def getClasses(numberOfBricks,numberOfSections):
  '''
  Diese Funktion benennt alle Klassen, die es geben kann für die Kombination (nOB,nOS).
  Es sind natürlich genauso viele, wie in getNumberOfClasses berechnet.
  
  return dictionary like
    getClasses(2,2) = {0:(0,0),1:(0,1),2:(1,0),3:(2,0),4:(0,2),5:(1,1)}
  (is in no way sorted)
  '''

  class myKey:
    '''
    zur Berechnung bauen wir einen Baum an Zuständen auf. die Zustände sind
    gleichzeitig die Namen der Klassen. Z.B.:
      Name = (0,2,0,0)
    bedeutet, dass im zweiten v-Intervall 2/numberOfBricks Prozent der maximal mög-
    lichen Wellen vorzufinden sind, während in allen anderen Intevallen nichts
    los ist. Bei (0,0,2,0) wäre das ganze im dritten v-Intervall, also gleich
    viel Prozent an maximal möglichen Wellen, aber diesmal schneller.
    '''
    def __init__(self,father,ith_child,noB=0,secs=0):
      '''
      wenn erstes Objekt, dann secs, mS != 0
      wenn kind, dann == 0 und alles von Vater erben
      
      secs: number of sections
      noB:   number of bricks
      '''
      # default class variables (for tree start element if no father exists)
      self.name_book_ = {}
      self.name_ = [0] * secs
      self.name_ = tuple(self.name_)
      self.secs_ = secs
      self.noB_ = noB
      
      self.father_ = None
      self.children_ = None
            
      if secs==0: # child case:
        self.father_ = father
        self.name_book_ = self.father_.name_book_
        self.name_ = [el for el in self.father_.name_]
        self.name_[ith_child] += 1
        self.name_ = tuple(self.name_)
        self.secs_ = self.father_.secs_
        self.noB_ = self.father_.noB_
      
      if self.name_ not in self.name_book_.values(): # this state was not found before:
        self.name_book_[len(self.name_book_)] = self.name_
      if self.noB_ > np.sum(self.name_): # did not reach last level. continue with next level.
        self.children_ = [myKey(self,i) for i in range(self.secs_)]
        
    def getDict(self):
      print("Received {} classes.".format(len(self.name_book_)))
      return self.name_book_.copy()
  
  # Note: if a is the dictionary, you can get the backward map as b = {a[el]: el for el in a}
  return myKey(None,1,numberOfBricks,numberOfSections).getDict() # The first two arguments None and 1 could be anything.


def repairBricks(NNClass,NBricksOld,NBricksNew):
  '''
  return value may be of type np.array. Doesn't effect stuff outwards. Makes it here more convenient.
  '''
  retValue = [int(round(el*NBricksNew/NBricksOld)) for el in NNClass]
  if sum(retValue) > NBricksNew: # occurs seldom.
    print("NNClass = {},NBricksOld = {},NBricksNew = {}".format(NNClass,NBricksOld,NBricksNew))
    # Achtung: [2,8,0,2],12,4 -> [0.66,2.66,0,.66] -> [1,3,0,1] sum > 4!
    # können dort kürzen, wo der relative Fehler am kleinsten ist.
    print("Warning: Introduced errors due to insufficient NN-class conversion.")
    unrounded = np.array([el*NBricksNew/NBricksOld for el in NNClass])
    dismissedRetValue = np.array(retValue)
    indicesGT0 = np.array(np.where(dismissedRetValue>0)[0]) # this is guaranteed to be of len>0. looks a bit complicated, but does the job.
    relativeError = (dismissedRetValue[indicesGT0]-unrounded[indicesGT0]) / unrounded[indicesGT0] # may contain positive, as well as negative values. only positive are of interest. Neg. ones are related to a value that was rounded down, so further down-rounding would not make sense.
    relevantIndices = indicesGT0[np.where(relativeError>0)]
    errorList = sorted(np.unique(relativeError))
    currentErrorIndex = 0
    while sum(retValue) > NBricksNew:
      if errorList[currentErrorIndex] > 0:
        indicesToBeDeminished = np.array(relevantIndices[np.where(np.isclose(relativeError,errorList[currentErrorIndex]))[0]])[0]
        retValue[indicesToBeDeminished] -= 1
      currentErrorIndex += 1
  return retValue

def classify(list_of_dicts,n_bricks,n_sections,is_integer_interpretation,velocities,waves_counted_probabilistic,maxW=0):
  '''
  In this method, we sort all dictionaries in list_of_dicts to their respective class.
  There are different encodings, according to the users preferences:
    integer
      -> one-hot encoding
    probabilistic  NOT IMPEMENTED AND WILL NOT! this encoding might be used in REGRESSION TASK, still to be implemented completely.
      -> prob. distrib.
 
  Main goal of this function is to create the output stream (truth of a NN with which it is trained)
  
  For example: dict1 = {1.0: 3, 3.33: 1, 6.66: 0, 10.0: 0} -- relates - to - class -->  (4,0)
  '''
  
  assert is_integer_interpretation, "CAUTION: probabilistic behaviour for classification tasks has never been tested! If you want to continue, please comment this line and procede accordingly!"
    # especially you should have a look at what happens in basics.dictToArray().
  
  # get the maximum number of waves that occurs in this list of dicts.
  if maxW == 0:
    maxW = basics.maxwaves(list_of_dicts,waves_counted_probabilistic)
        
  if n_bricks == 0: # this is due to autoMode
    n_bricks = maxW
    print("Warning: Setting number of Bricks (command line argument B) to {}".format(maxW))
  # calculate how many Classes there exist
  NoC = getNumberOfClasses(n_bricks,n_sections)
  classMapping = getClasses(n_bricks,n_sections)
  backwardsMap = {classMapping[el]: el for el in classMapping}

  # hierin können wir die Verteilung der Input-Daten auf die existierenden Klassen schreiben.
  # aktuell nicht benötigt, bzw. keine Weiterverarbeitung.
  classDistribution = np.ndarray([NoC],dtype='int')*0
  #print(backwardsMap)

### Die metadaten enthalten beide versionen, p und i zeitlgleich als metaObject.pData_ und metaObject.iData_
  length = len(list_of_dicts)
  output = np.zeros([length,NoC]) # output[i] is in R^{NoC}, either a one hot encoding (all zeros, but one 1) or a probability distribution (adds to one).
  # WIR VERZICHTEN (aus Zeitgründen) vorerst auf die probabilistischen Daten (die nicht-one-hot-NNs)
  for i,el in enumerate(list_of_dicts):
    # convert dict to array, stateing the number of waves per wave speed section
    # Note: WavesPerSection holds integer values, to fit the name scheme of the classes.
    #       It is not normalized like it is for regression tasks.
    WavesPerSection = basics.dictToArray(el,n_sections, is_integer_interpretation,velocities)
    # stretch/shrink if not 'auto/direct' behaviour:
    if n_bricks != maxW:
      WavesPerSection = repairBricks(WavesPerSection,maxW,n_bricks)
    #print(WavesPerSection)
    className = tuple(WavesPerSection)
    classNumber = backwardsMap[className]
    classDistribution[classNumber] += 1
    output[i][classNumber] = 1
  
  
  return output,maxW,NoC,classMapping


def autolabel(rects, ax):
  # Get y-axis height to calculate label position from.
  (y_bottom, y_top) = ax.get_ylim()
  y_height = y_top - y_bottom
  for rect in rects:
    height = rect.get_height()
    # Fraction of axis height taken up by this rectangle
    p_height = (height / y_height)
    # If we can fit the label above the column, do that;
    # otherwise, put it inside the column.
    if p_height > 0.95: # arbitrary; 95% looked good to me.
      label_position = height - (y_height * 0.05)
    else:
      label_position = height + (y_height * 0.01)
    ax.text(rect.get_x() + rect.get_width()/2., label_position,'%d' % int(height),ha='center', va='bottom')

def investigate(model,valData,step_size,noB,scene_mapping):
  #import scenario as sc
  x_validation,y_validation = valData
  classMap,velIds_train,velIds_validation,groundtruths,maxW = scene_mapping
  
  print(velIds_validation.shape)
  #list(map('{:.3f}%'.format,y_validation[num]))
  print("We will have a look on {} more or less random results.".format(len(np.arange(0,x_validation.shape[0],step_size))))
  for num in range(0,x_validation.shape[0],step_size):
    processed_data = model(x_validation[num:num+1])
    #print(processed_data)
    class_i = np.argmax(processed_data[0])
    reference_class = np.argmax(y_validation[num])
    #print("")
    print("NN Class predicted: {}, with {:.1f}% certainty.".format(class_i,100*np.max(processed_data[0])))
    P_ref = processed_data[0][reference_class]
    print("reference solution: {} (would have had {:.1f}% probability).".format(reference_class,P_ref*100))
    #print("control sum: {}".format(sum(processed_data[0])))
    #print(processed_data[0])
    #print("y_validation: \n {}".format(y_validation[num]))
    predictedClassName = classMap[class_i]
    TrueClassName = classMap[reference_class]
    originalDicct = groundtruths[velIds_validation[num][0]]
    wave_info = originalDicct,maxW
    if reference_class != 0 or class_i != 0:
      showClass(predictedClassName,noB,TrueClassName,wave_info)

def investigate_new(model,network_info,step_size=1):
  """
    this function compares a models predictions to the validation data given in the network information
    
    model: the functional KERAS model.
    network_info: dict or old tuple
                  the parameters that were used to train this model. (training data is 'lost' but can be tracked back most probalby.)            
  """
  #import scenario as sc
  assert isinstance(network_info,dict), "Don't use the new investigate routine!"
    # we have everything in the new fashion:
  x_validation,y_validation = network_info['valData']
  classMap = network_info['SMap']
  maxW = network_info['maxWaves']
  velIds_validation = network_info['gtruth_id']
  groundtruths = network_info['gtruth']
  noB = network_info['NBricks']
  
  print(velIds_validation.shape)
  #list(map('{:.3f}%'.format,y_validation[num]))
  print("We will have a look on {} more or less random results.".format(len(np.arange(0,x_validation.shape[0],step_size))))
  for num in range(0,x_validation.shape[0],step_size):
    processed_data = model(x_validation[num:num+1])
    #print(processed_data)
    class_i = np.argmax(processed_data[0])
    reference_class = np.argmax(y_validation[num])
    #print("")
    print("NN Class predicted: {}, with {:.1f}% certainty.".format(class_i,100*np.max(processed_data[0])))
    P_ref = processed_data[0][reference_class]
    print("reference solution: {} (would have had {:.1f}% probability).".format(reference_class,P_ref*100))
    #print("control sum: {}".format(sum(processed_data[0])))
    #print(processed_data[0])
    #print("y_validation: \n {}".format(y_validation[num]))
    predictedClassName = classMap[class_i]
    TrueClassName = classMap[reference_class]
    originalDicct = groundtruths[velIds_validation[num][0]]
    wave_info = originalDicct,maxW
    if reference_class != 0 or class_i != 0:
      print("\npredicted Class Name: {}".format(predictedClassName))
      print("number of bricks: {}".format(noB))
      showClass(predictedClassName,noB,TrueClassName,wave_info)

def showClass(name,noOfBricks,truth=False,originalWaves=False):
  '''
    make a bar plot
    showing the specific class that is represented via its unique name
    The information noOfBricks is needed, since it is not contained in the class name.
    
    truth is optional, if used, it must be the true class name. If it is given,
    the produced plot compares the guess and the truth next to each other.
  '''
  
  granularity = 100. / noOfBricks
  sections = len(name)
  bar_heights = [el*granularity for el in name]
  labels = ["[{:.3}, {:.3})".format(i/sections*10. + (sections-i)/(sections),(i+1)/sections*10.+(sections-(i+1))/(sections)) for i in range(sections)]
  x = np.arange(len(labels))
  print("x is {}".format(x))
  width=0.99 # (default = 0.8)
  fig, ax = plt.subplots()
  ax.set_ylim(0.,100.)
  if truth:
    if type(truth) != type(name):
      raise Exception("truth must be a tuple of size {}".format(len(name)))
    elif len(truth) != len(name):
      raise Exception("truth must be a tuple of size {}".format(len(name)))
    width = width/2
    rects = ax.bar(x,bar_heights,1.9*width,label='guessed distribution',color='blue')#-width/2
    true_activity = [el*granularity for el in truth]
    true_rects = ax.bar(x,true_activity,2*width,label='ground thruth',color='green',alpha=0.4)# +width/2
    autolabel(rects,ax)
    autolabel(true_rects,ax)
  else:
    rects = ax.bar(x,bar_heights,width,label='fibre type distribution',color='blue')
    autolabel(rects,ax)
  if originalWaves:
    dicct,maxW = originalWaves
    velocities = np.array(sorted([key for key in dicct.keys()])) # every vel is unique and they are different in pairs
    numberV_i = [dicct[vel]/maxW*100 for vel in velocities]
    print("displayed velocities: {}".format(velocities))
    # map velocities from [1,10] to [-0.5, sections-0.5]
    mappedVelocities = ((velocities-1)/9*sections)-0.5
    peaks = ax.bar(mappedVelocities,numberV_i,width/10,label='original wave distribution',color='grey')
    
  ax.set_ylabel('muscle fibre activity [\%]')
  ax.set_xticks(x)
  ax.legend()
  #ax.bar_label(rects,padding=3)
  ax.set_xticklabels(labels)
  ax.set_xlabel('Fibre sEMG velocity [m/s]')
  fig.tight_layout()
  plt.show()
 
