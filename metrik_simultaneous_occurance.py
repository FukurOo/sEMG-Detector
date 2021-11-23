#Beschreibung: Metrik misst wie oft welche Geschwindigkeit mit einer manuell festgelegten
# zu prüfenden Geschwindigkeit in einem Datensatz auftritt
# Nur sinnvoll bei Daten, die von creator_002 erzeugt wurden

import pickle
import numpy as np
import sys
import matplotlib.pyplot as plt
import itertools


#Variablen Datensatzes erzeut von data_creator_00x übergeben
creator = str(sys.argv[1])
n_speeds = str(sys.argv[2])
n_waveLengths = str(sys.argv[3])
#@TODO: was ist der grow_factor? eventuell variabel machen
grow_factor = 4

pruef_speed = str(sys.argv[4])
pruef_speed = float(pruef_speed)

#je nach creator den passenden Dateinamen nutzen
if creator == '1':
    input = "DATA/creator_001/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths.pickle"
else:
    input = "DATA/creator_002/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths___grow_factor_"+str(grow_factor)+".pickle"


#zum Datensatz gehörendes pickle Dokument wird geladen
f = open(input, 'rb')
daten = pickle.load(f)


#Scenarien extrahieren in denen Geschwindigkeiten mit der zu prüfenden Geschwindigkeit pruef_speed gemeinsam auftreten
#die zu prüfende Geschwindigkeit wird dabei entfernt, oder als Vergleich behalten?
number_of_simultaneous_occurance = []
for i in range(len(daten)):
    u = daten[i].meta_.get_vecs()[2]
    if pruef_speed in u:
#        u.remove(pruef_speed)
        number_of_simultaneous_occurance.append(u)

#Liste 1-dimensional machen
number_of_simultaneous_occurance = list(itertools.chain(*number_of_simultaneous_occurance))


#Histogramm
#@ToDo: mehr x-Werte anzeigen für bessere Lesbarkeit der Geschwindigkeiten
#@ToDo: ggf statt der 4 in i/4 die #speeds-1 als Wert nehmen
n, bins, patches = plt.hist(number_of_simultaneous_occurance, bins = [i/4 for i in range(42)])
#Ausgabe der Häufigkeit des gemeinsamen Auftretens aller Geschwindigkeiten im Terminal
print('[Geschwindigkeit,Häufigkeit mit {} auftreten]'.format(pruef_speed))
speed_occ = []
for i in range(len(bins)-1):
    if n[i] != 0:
        speed_occ.append([bins[i], n[i]])
print(speed_occ)
plt.title("{}".format(input))
plt.xlabel("Geschwindigkeit in m/s")
plt.ylabel("Häufigkeit mit {} auftreten".format(pruef_speed))
plt.show()
