#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Di Nov 09 11:30:00 2021

@author: Maximilian Hack, Lilian Cathérine Lepère, Lisa 
 
"""

#
# Dieses Modul stellt eine Verfeinerung des data_creator_001 dar.
# Gewschwindigkeiten werden von nun an aus mehreren Intervallen ausgewählt, für die gilt:
# [x,y], x,y in [1,...,10] und x < y
# Zudem werden die Intervallgrenzen mit einer bestimmten Wahrscheinlichkeit ausgewählt

import scenario as sc
import data_creative_tools as dct
import numpy as np
import pickle
import sys
import random


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

#<------------------------------------------------------------------------------------>
scenarios = []                                                   # enthält alle erstellten Szenarien
intervallList = [i for i in range(1,11)]                         # alle Zahlen, die als obere, bzw. untere Intervallgrenze infrage kommen
weightsList = (1,1.25,1.5,1.75,2,2,1.75,1.5,1.25,1)              # Wahrscheinlichkeiten, mit denen eine Intervallgrenze ausgewählt werden soll 

for i in range(n_waveLengths) :
    a = 0
    b = 0
    while(a == b):
        randomIntervall = random.choices(
            intervallList, weights=weightsList, k=2)
        a = randomIntervall[0]
        b = randomIntervall[1]
        if (a<b) :
            start = a
            end = b
        start = b
        end = a
    scenarios.append(dct.get_scenarios_with_fixed_wavelength(n_speeds,lengths[i],world,1/2048,16,0.075,0.5, start, end))
#<------------------------------------------------------------------------------------>



# Benennung der erzeugten Datei
file_name = "DATA/advanced_data_creator/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths.pickle"
# Speicherung der erstellten Szenarien als pickle-Datei
pickle.dump(scenarios, open(file_name, "wb"))