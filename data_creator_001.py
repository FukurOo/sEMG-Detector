#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:33:01 2021

@author: kraemer

data_creator_001.py
"""

#
# Dieses Programm erzeugt Daten.
# Die erzeugten Daten unterscheiden sich in
#  - Anzahl verschiedener Geschwindigkeiten auf [1,10]
#  - Anzahl unterschiedlicher Wellenformen (also Wellenlänge) auf [0.01,0.10]
#

import scenario as sc
import data_creative_tools as dct
import numpy as np
import pickle
import sys

n_speeds = int(sys.argv[1])
# empfohlene Werte n in [2,30]

n_waveLengths = int(sys.argv[2])
# empfohlene Werte n in [10,30]


# segment_length = 0.075 is default
# number of electroids 16 is default
# sEMG frequency 1/2048 is default
# critical value for signal signal strength interpretation 0.5 is default
#
# defaultwerte werden unten trotzdem angegeben, dass die Möglichkeit zur Änderung sichtbar bleibt 

lengths = np.linspace(0.01,0.10,n_waveLengths)

world = sc.Organizer()

# returned object will hold artificial sEMG-data as well as raw meta-data, which describes the contained data.
scenarios = [dct.get_scenarios_with_fixed_wavelength(n_speeds,lengths[i],world,1/2048,16,0.075,0.5) for i in range(n_waveLengths)]

# get unique and descriptive name for the file
file_name = "DATA/creator_001/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths.pickle"
# save it
pickle.dump(scenarios, open(file_name, "wb"))

