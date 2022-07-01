#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:55:11 2021

@author: kraemer
"""
# %%
# Dies ist das zentrale programm, um NNs zu erzeugen
# ========================================================
#
# Es können verschiedene Parameter / Einstellungen verändert / ausgesucht werden.
#
# NN-Architektur
# --------------
# 
# Art des y-trains also der ground truth: integer oder probabilistic
# ------------------------------------------------------------------
# 
# eindeutige oder mehrdeutige Klassen (granular oder direkt)
# ----------------------------------------------------------
#   - innerhalb des granularen Ansatzes:
#      - Anzahl Sektionen
#      - Granularität
#     (- damit verbunden die Zahl der resultierenden Klassen)
#   - innerhalb des direkten Ansatzes
#      - Anzahl Sektionen
#      - (noch offen)
# 
# Anzahl Epochen
# --------------
#
#
# Größe der Trainingsdaten
# ------------------------
#   - entsprechend des Dateinamens, der die Daten enthält.
#   - müssen zuvor mit data_creator_xyz.py erzeugt werden.
#
# %%
'''
Ausführung des Programms:
 $ python3 regression.py <arg list>

 Um allerdings die Ausgabe auf das Terminal zu dokumentieren, ist man momentan
 auf hacks wie tee angewiesen:
 $ python3 main.py <arg list> 2>&1 | tee temporary_output_file
 
 Dadurch sieht man live im terminal wo das Programm ist UND befüllt eine Ausgabe-
 datei. Mit
 $ mv <ouftfile> <DEST_PATH> (so getan in execute-me.sh)
 wird per post processing die obige Ausgabedatei an den richtigen Platz verschoben.
 Die Datei execute-me.sh wird dafür für jedes trainierte NN neu geschrieben und
 enthält einen eindeutigen Pfad, wo das Ergebnis gespeichert wird.
 
 Ausführung und Verschieben wird in script.sh exemplarisch kombiniert. Analog
 kann sehr einfach ein skript erstellt werden um Parameter-Studien o. ä. durch-
 zuführen.
'''
# %% 


# %%
import argument_stuff as USER
import file_stuff as MMRY
import training_stuff as NN
import pickle

# in the following dict, there are all the informations we need, to select the options
# which we are going to apply to a new neural network.

training_options = USER.getRegressionArgumentList()
USER.process(training_options)

output_path = MMRY.preProcess(training_options,'REG_TEST/')

network = NN.NeuralNetwork(training_options)

#pic_shape, training_data, validation_data, number_of_classes, 
network.getTrainingData()

#history,model 
network.startTraining()#pic_shape,training_data,validation_data,number_of_classes,

#NN.investigate(model,validation_data,2000, training_options['B'] ,scene_mapping)

#MMRY.postProcess(output_path,history,model,(scene_mapping,validation_data))
network.saveModel(output_path)




network.saveMemorySpace(radical=True) # this deletes the original model and training data!

print(network)

with open(output_path+'/network_info.pickle', 'wb') as pfile:
  #print(network.__dict__)
  pickle.dump(network.__dict__, pfile,protocol=pickle.HIGHEST_PROTOCOL)














