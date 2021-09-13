#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 17:21:48 2021

@author: kraemer
"""
#
# Dieses Programm erzeugt Daten.
# Die Daten unterscheiden sich in
#  - Anzahl verschiedener Geschwindigkeiten auf [1,10]
#  - Anzahl unterschiedlicher Wellenformen (also Wellenlänge) auf [0.01,0.10]
#
# ..müsste ich hier noch den Datensatz irgendwie charakterisieren?
# .. das sollte aber auch nachträglich funktionieren.
#
#


import scenario as sc
import data_creative_tools as dct
import numpy as np
import pickle
import sys


n_speeds = 2#10#int(sys.argv[1])
n_waveLengths = 1#0#int(sys.argv[2])

# segment_length = 0.075 is default
# number of electroids 16 is default
# sEMG frequency 1/2048 is default

world = sc.Organizer()
# wir erstellen jetzt eine Reihe an Szenarien mit unterschiedlichen wellenlängen.
lengths = np.linspace(0.01,0.10,n_waveLengths)

# jedes Szenario hat eine Anzahl unterschiedlicher Geschwindigkeiten:
# is an array of arrays of objects:
# hat den Vorteil, dass ich die Info nicht verlieren kann, wie viele verschiedene wave_lengths drin sind.
scenarios = [dct.get_scenarios_with_fixed_wavelength(n_speeds,lengths[i],world,1/2048,16,0.075,0.5) for i in range(n_waveLengths)]

all_scenarios = []
for scene in scenarios:
  for el in scene:
    all_scenarios.append(el)

file_name = "RawData_"+str(n_speeds)+"_V_and_"+str(n_waveLengths)+"_Lambda"

### TODO: überlagern! deshalb hab ich mir doch den ganzen Scenario-Objekt-Aufwand gegeben! Außerdem macht es das NN robuster!

pickle.dump(all_scenarios, open(file_name, "wb"))
  #pickle.dump(scenarios, open("myobject_AofAs", "wb"))
#else:
#  all_scenarios = pickle.load(open("trial 10x10", "rb"))
#  # is an array of objects:
#  # muss mir irgendwie merken, wie viele verschiedene wave_lengths drin sind.
#all_scenarios[0].compare(10,10,*all_scenarios[1:])
#
#all_scenarios[0].compare(0,0,*all_scenarios[1:])




# in folgendem sollen sich die wellen unterscheiden:
 
# Geschwindigkeit v in [v_min,v_max]

# Wellenlänge lambda in [lambda_min, lambda_max]

# Anzahl N in {0, 1, 2, 3} (3 kann es bei default detection criterion nur geben, wenn lambda <(=?) 5 ist.)

# Abstand der Wellen; 

# Amplitude a in [eps,1] ?   sollte besser getrennt untersucht werden, was das für einen effekt hat.. 
























