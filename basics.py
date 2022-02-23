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


def getSectionForVelocity(v,borders):
  # determine interval number that v belongs to:
  return borders.searchsorted(v - (borders[1]-borders[0])) + int(v in borders) - int(v==borders[0] or v==borders[-1])

def dictToArray(dicct,numberOfSections,integer_behaviour,velocities): # diese funktion zählt die anzahl an wellen in den einzelnen sektionen. das ist also auch für die regression sinnvoll, da auch dort y_training und y_validation erzeugt werden müssen.
  '''
  Function converts the single amounts of waves for each velocity to an array with combined numbers per section.
  This is done for both, integer and probabilistic behaviour.
    * integer: f_i : \R^{N_v} --> \R^{N_s}
    * prob.:  f_p: DPD^{N_v} --> DPD^{N_s}, where DPD is some discrete prob. distr. (with at least one state).
  
  integer_behaviour: true is default.
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
  
  ToDo: Implement probabilistic functinality.
  
  ToDo: get information of existing velocities! For now, we have to assume
  that 1.0 is the smallest, and 10.0 is the largest velocity. If this is not
  the case, ALL INTERVALS ARE PLACED WRONG!
  '''
  
  if not integer_behaviour:
    raise Exception("ToDo: Implement probabilistic functinality!")
  
  if velocities:
    vmin = velocities[0]
    vmax = velocities[-1]
  else:
    vmin = 1.0
    vmax = 10.0
  
  retVal = [0]* numberOfSections
  borders=np.linspace(vmin,vmax,numberOfSections+1)
  #if classification task or integer
  for vel in dicct.keys():
    s_i = getSectionForVelocity(vel,borders)
    if integer_behaviour:
      # add waves to section: (Note: for probabilistic behaviour (within the classification task) this must be some kind of a DPD addition!)
      retVal[s_i] += dicct[vel]
    else: # probabilistic regression task:
    # this might crash for classification tasks or even run through, but produce wrong results. Could be right as well. need to check if ever used with classification task!      
      for event in dicct[vel].keys():
        retVal[s_i] += float(event)*dicct[vel][event]#  Add all discrete expected values (Erwartungswerte) of the velocities(E(v)). this is then the expected value of the associated Interval. (in this formula, we skip the explicit interim result E(v) and add to E(I) directly.)
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