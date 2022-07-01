#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:42:16 2021

@author: kraemer

basic functions that deal with:
 * meta-information,
 * ..?
"""
import numpy as np

def maxwaves(vlist,waves_counted_probabilistic = True):
  '''
  this function searches for the maximum amount of concurrently ocurring waves, regardless of their velocitiy
  
  there could be a faster version for "pure" Data (such as data produced by data_creator001). However we're not interested in speed here, and it's also a rare occasion..
  
  Parameters:
    vlist:                      list of dicts
                                every dict is the meta information to a specific picture. meaning, it could be a integer or probabilistic point of view velocity-count interpretation.
    waves_counted_probabilistic:  optionalb, bool
                                gibt an, auf welche Art und Weise die Daten interpreteiert werden sollen.
                                Achtung: hängt vom Datenpfad ab, nicht von is_integer_interpretation!
                              
  '''

  assert isinstance(vlist[0],dict), "Can't work with this argument!"
  maxW = 0
  length = len(vlist)
  count = np.ndarray([length],dtype=np.int32)
  for i in range(length):
    count[i] = 0
    for kel in vlist[i].keys():
      if waves_counted_probabilistic: # probabilistic wave-counting leads to a more complex data structure here:
        count[i] += int(max(vlist[i][kel].keys()))
      else: # integer wave-counting
        count[i] += int(vlist[i][kel])
    #if count[i] > maxW:
    #  maxW = int(count[i])
  maxW = count.max()
  return maxW

def getSectionForVelocity(v,borders):
  # determine interval number that v belongs to:
  if v > borders[-1] or v < borders[0]:
    print("Warning: velocity out of bounds!")
  return borders.searchsorted(v - (borders[1]-borders[0])) + int(v in borders) - int(v==borders[0] or v==borders[-1])

def dictToArray(dicct,numberOfSections,integer_behaviour,velocities): # diese funktion zählt die anzahl an wellen in den einzelnen sektionen. das ist also auch für die regression sinnvoll, da auch dort y_training und y_validation erzeugt werden müssen.
  '''
  Function converts the single amounts of waves for each velocity to an array with combined numbers per section.
  This is done for both, integer and probabilistic behaviour.
    * integer: f_i : \R^{N_v} --> \R^{N_s}
    * prob.:  f_p: DPD^{N_v} --> DPD^{N_s}, where DPD is some discrete prob. distr. (with at least one state).
  
  integer_behaviour: this parameter determines how addition of waves works. (aka DPDs)
  dicct: the amount of waves per contained velocity. Values are either \R^{N_v}
         or DPD^{N_v}, depending on use case (behaviour)
         Note: There are at max N_v velocities, not all of them are contained in every dicct. Also, N_v is unknown!  N_v will be made known through a sorted list of all velocities.
  numberOfSections: Here N_s. (or number of intervals, if  regression task)
    
  In this fuction, we have to decide, what happens if a velocity is exactly on a border. However,
  this is probably only a convention and does not effect the NNs qualitative behaviour.
  The implemented convention is:
    all intervals are half open:                '[)',
    except for the last one, which ist closed:  '[]'.
  This means, "border-velocities" usually get encounted for by the interval on its right side.
  Only on the last (upper) border, the border-velocity has to be counted to the left interval.
  For example, with (v_1,...v_6) = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), and N_s = 5,
  the situation looks like:
    '[v_1), [v_2), [v_3), [v_4), [v_5,v_6]'
    
  N_s (usually is, but) does not have to be smaller than N_v.
  For regression tasks, N_s is by default N_v. Only if specified explicitly, it will be something else.
  '''
  assert numberOfSections > 0, 'Odd case. probably not implemented well.'
  
  if velocities:
    vmin = velocities[0]
    vmax = velocities[-1]
  else:
    vmin = 1.0
    vmax = 10.0
  borders=np.linspace(vmin,vmax,numberOfSections+1)
  retVal = [0]* numberOfSections
  
  if integer_behaviour:
  # integer point of view: we are working with destinct states (not with events)
    for vel in dicct.keys():
      s_i = getSectionForVelocity(vel,borders)
      if integer_behaviour:
        # add waves to section:
        retVal[s_i] += dicct[vel]
    return retVal
  else:
    # probabilistic point of view: we are working with DPDs (each discrete event has its probability)
    tmp_dpd = [DPD()]*numberOfSections
    for vel in dicct.keys():
      s_i = getSectionForVelocity(vel,borders)
      tmp_summand = DPD(dicct[vel])
      tmp_dpd[s_i] += tmp_summand
    for i in range(numberOfSections):
      expected_value = 0
      for event,probab in tmp_dpd[i].items():
        expected_value += event*probab
      retVal[i] = expected_value
    #print(retVal)
    return retVal

### TODO: wo wurde diese Funktion benutzt? ist sie in das Scenario-file gerutscht, wo sie eventuell hingehört?
def P_N_waves_are_there(b):
  k = [el for el in b.keys()]
  if len(k)==0:
    print("size error in function f")
  t_new={}
  for speed in b[k[0]].keys():
    t_new[speed] = b[k[0]][speed]
  #print(t_new)
  for i in range(1,len(k)):
    #for j in range(i):
    t_old = t_new.copy()
    t_new = {}
    for m in b[k[i]].keys():
      for n in t_old.keys():
        if (m+n) in t_new.keys():
          t_new[m+n] += b[k[i]][m]*t_old[n]
        else:
          t_new[m+n] = b[k[i]][m]*t_old[n]
    #print(t_new)
  return t_new


# die folgende Klasse hätte auch in meta_stuff verwendet werden können. Jetzt drauf gesch... Verwendung aber in training_stuff und dergleichen.
class DPD(dict):
  '''
    we use DPDs to describe how many waves of a specific velocity are in a domain.
    to superpose two velocities we need to add two DPDs. This class allows us to do this very handy.
    There's also a __mul__ version..
  '''
  def __init__(self, *args, **kwargs):
    self.update(*args, **kwargs)
  
  def __getitem__(self, key):
    val = dict.__getitem__(self, key)
    #print('GET', key)
    return val
  
  def __setitem__(self, key, val):
    #print('SET', key, val)
    dict.__setitem__(self, key, val)
  
  def __repr__(self):
    dictrepr = dict.__repr__(self)
    return '%s(%s)' % (type(self).__name__, dictrepr)
        
  def update(self, *args, **kwargs):
    #print('update', args, kwargs)
    for k, v in dict(*args, **kwargs).items():
      #if not(isinstance(k,float) or isinstance(k,int) or isinstance(k,np.float64) or isinstance(k,np.int64)):
      #  print("typ {} ist {}".format(type(k),k))
      assert (isinstance(k,float) or isinstance(k,int) or isinstance(k,np.float64) or isinstance(k,np.int64)), "Not convertable into class DPD!"
      self[k] = v
      
  def __add__(self,other):
    assert isinstance(other, DPD), "Addition task unknown"
    if len(self) == 0:
      return other
    elif len(other) == 0:
      return self
    else:
      newDict = {}
      for i in self.keys():
        for j in other.keys():
          if i+j not in newDict.keys():
            newDict[i+j]=0
          newDict[i+j] += self[i] * other[j]
      # init and return new DPD:
      #print(newDict.values())
      assert np.isclose(np.sum(np.array([f for f in newDict.values()])),1), "DPD sum != 1. {}".format(np.sum(np.array(newDict.values())))
      return DPD(newDict)
  
  def __mul__(self,other):
    assert False, "Do you realy want to use this???" # Function works just fine, but there should not be a use for this!
    assert isinstance(other, DPD), "Multiplication task unknown"
    # init dict to initialize new DPD:
    newDict={}
    for i in self.keys():
      for j in other.keys():
        if (i*j) not in newDict.keys():
          newDict[i*j] = 0
        newDict[i*j] += self[i] * other[j]
    # init and return new DPD:
    assert np.isclose(np.sum(np.array(newDict.values())),1), "DPD sum != 1. {}".format(np.sum(np.array(newDict.values())))
    return DPD(newDict)