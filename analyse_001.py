#from scenario import *
import pickle
import numpy as np
import sys

#Name des Datensatzes erzeut von data_creator_001 übergeben
input = str(sys.argv[1])

#zum Datensatz gehörendes pickle Dokument wird geladen
f = open(input, 'rb')
daten = pickle.load(f)

daten = np.array(daten)

#Scenarien werden einzeln ausgegeben
for i in range(np.shape(daten)[0]):
    for j in range(np.shape(daten)[1]):
        daten[i,j].analyse()
