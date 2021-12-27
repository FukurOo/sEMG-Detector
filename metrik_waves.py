#Metrik misst wie viele Wellen in einem Datensatz Auftreten

import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt


#Variable x des Datensatzes erzeut von data_creator_00x übergeben
#oder des ganzen Dateiname eines Datensatzes
#Was übergeben wurde wird in der nächsten if-Abfrage überprüft
creator = str(sys.argv[1])

#Dateiname oder 'Kennzahlen'
if len(creator) > 4:
    input = creator
    #@TODO: n_speeds extrahieren

else:
    creator = creator
    n_speeds = str(sys.argv[2])
    n_waveLengths = str(sys.argv[3])
    #@TODO: grow_factor eventuell variabel -> grow_factor hat den Default-Wert 4.
    grow_factor = 4

#je nach creator den passenden Dateinamen nutzen
    if creator == '1':
        input = "DATA/creator_001/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths.pickle"
    else:
        input = "DATA/creator_002/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths___grow_factor_"+str(grow_factor)+".pickle"


#zum Datensatz gehörendes pickle Dokument wird geladen
f = open(input, 'rb')
daten = pickle.load(f)

daten = np.array(daten)

#Format des arrays in dem die scenarios gespeiechert sind prüfen und 1-dimensional machen
if daten.ndim != 1:
    daten = daten.flatten()

#extrahieren der Geschwindigkeiten aus den Meta-Daten und speicher in speeds
waves = []
for d in daten :
    #Ein Scenario besteht aus einer Liste, in der die Anzahl an vorhandenen Wellen
    # während eines Zeitschritts in einem dictionary gespeichert sind
    for j in range(len(d.meta_.iData_)):
        keys = d.meta_.iData_[:][j].keys()
        for i in keys:
            occurance = d.meta_.iData_[:][j].get(i)
            for k in range(occurance):
                waves.append(i)

print(daten[10].meta_.iData_[:][0].keys())
print(daten[10].meta_.iData_[:][0].get(1))

#Histogramm
#@ToDo: mehr x-Werte anzeigen für bessere Lesbarkeit der Geschwindigkeiten
#@ToDo: ggf statt der 29 in i/29 die #speeds-1 als Wert nehmen
n, bins, patches = plt.hist(waves, bins = [i/29 for i in range(292)])
#Ausgabe der Häufigkeit der Wellen aller auftretenden Geschwindigkeiten im Terminal
print('[Geschwindigkeit, Anzahl Wellen]')
speed_occ = []
for i in range(len(bins)-1):
    if n[i] != 0:
        speed_occ.append([bins[i], n[i]])
print(speed_occ)
plt.title("{}".format(input))
plt.xlabel("Geschwindigkeit in m/s")
plt.ylabel("Anzahl Wellen")
plt.show()
