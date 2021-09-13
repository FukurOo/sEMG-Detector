#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 10:55:11 2021

@author: kraemer
"""
# %%
# ich will EIN zentrales programm, das mir die NNs erzeugt
# ========================================================
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
# [DAS WILL ICH HIER ENTKOPPELN. SOLL 1 MAL ERZEUGT WERDEN UND DANN ALLEM DIENEN]

# %%
# Ausführung des Programms:
# um die Ausgabe auf das Terminal zu dokumentieren, bin ich vorerst auf tee angewiesen:
# $ python3 main.py <arg list>   
# dadurch sehe ich live im terminal wo das Programm ist UND befülle eine Ausgabedatei.
#
# mit $ mv <ouftfile> <DEST_PATH>
# könnte ich post processing mäßig per shell script dann die Datei an den richtigen Platz verschieben.
# %% 


# %%
import argument_stuff as USER
import file_stuff as MMRY
import training_stuff as NN

# in the following dict, there are all the informations, we need to select the options
# which we are going to apply to a new neural network.
training_options = USER.processArguments()

data_file,output_path = MMRY.preProcess(training_options,'MODELS/')

pic_shape, training_data, validation_data, number_of_classes = NN.getTrainingData(data_file,training_options)

history,model = NN.trainNeuralNetwork(pic_shape,training_data,validation_data,number_of_classes,training_options)

MMRY.postProcess(output_path,history,model)

















