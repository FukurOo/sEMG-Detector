#Beschreibung: misst wie oft eine Geschwindigkeit in einem Datensatz insgesamt vorkommt

import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt


#Name des Datensatzes erzeut von data_creator_00x übergeben
input = str(sys.argv[1])

#zum Datensatz gehörendes pickle Dokument wird geladen
f = open(input, 'rb')
daten = pickle.load(f)

daten = np.array(daten)

#bei 001 gibts pro scenario genau eine Geschwindigkeiten
#bei 002 kann es bis zu (grow_factor_4?) 4 pro scenario geben


#Format des arrays in dem die scenarios gespeiechert sind prüfen und 1-dimensional machen
if daten.ndim != 1:
    daten = daten.flatten()


speeds = []
for i in range(np.shape(daten)[0]):
    #get_vecs()[2] ist accumulated_speeds
    speeds.append(daten[i].meta_.get_vecs()[2][0])


#Histogramm
#@ToDo: mehr x-Werte anzeigen für bessere Lesbarkeit der Geschwindigkeiten
#@ToDo: ggf statt der 4 in i/4 die #speeds-1 als Wert nehmen
n, bins, patches = plt.hist(speeds, bins = [i/29 for i in range(292)])
#Ausgabe der Häufigkeit aller Geschwindigkeiten im Terminal
print('[Geschwindigkeit,Häufigkeit]')
print([[bins[i], n[i]] for i in range(len(bins)-1)])
plt.title("{}".format(input))
plt.xlabel("Geschwindigkeit in m/s")
plt.ylabel("Häufigkeit")
plt.show()
