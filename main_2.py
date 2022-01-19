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
 $ python3 main.py <arg list>

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
import training_stuff_2 as NN
import numpy as np
import matplotlib.pyplot as plt

# in the following dict, there are all the informations we need, to select the options
# which we are going to apply to a new neural network.
training_options = USER.processArguments()

data_file,output_path = MMRY.preProcess(training_options,'MODELS/')

pic_shape, training_data, validation_data, number_of_classes, scene_mapping  = NN.getTrainingData(data_file,training_options)

x_validation, y_validation = validation_data[0], validation_data[1]
#454 training_stuff: gibt Klasse an
reference_class = [np.argmax(y_validation[i]) for i in range(0,x_validation.shape[0])]
for i in range(112800):
	if reference_class[i] != 0:
		print(i, reference_class[i], (np.shape(x_validation[i]), x_validation[i]), y_validation[i],scene_mapping[2][i])
		break
print('refrence_class[3094] ', reference_class[10], reference_class[3094])

#Histogramm
#@ToDo: mehr x-Werte anzeigen für bessere Lesbarkeit der Geschwindigkeiten
#@ToDo: ggf statt der 4 in i/4 die #speeds-1 als Wert nehmen
n, bins, patches = plt.hist(reference_class, bins = [i for i in range(number_of_classes+1)])
#Ausgabe der Häufigkeit aller Geschwindigkeiten im Terminal
#@ToDo: nur Geschwindigkeiten ausgeben, die auch wirklich vorkommen
print('[Klasse,Häufigkeit]')
print([[bins[i], n[i]] for i in range(len(bins)-1)])
print(np.sum(n))
#training_stuff -> ges Anzahl an n = 22108
#training_stuff_2 -> ges. Anzahl an n = 22109

#training_stuff_2 uses 100% of data for validation -> each scenarios' class is represented in Histogramm 
#NO!!!
#There are only 3384 scenarios in  
#110544


plt.title("{}".format('Verteilung an NN-Klassen'))
plt.xlabel("Klasse")
plt.ylabel("Häufigkeit")
plt.show()


'''
NN_2 = [[0, 1671.0], [1, 3258.0], [2, 1296.0], [3, 288.0], [4, 71.0], [5, 40.0], [6, 37.0], [7, 37.0], [8, 21.0], [9, 2.0], [10, 0.0], [11, 0.0], [12, 0.0], [13, 0.0], [14, 0.0], [15, 0.0], [16, 0.0], [17, 1.0], [18, 2.0], [19, 0.0], [20, 10.0], [21, 1.0], [22, 0.0], [23, 0.0], [24, 36.0], [25, 8.0], [26, 1.0], [27, 1.0], [28, 1.0], [29, 64.0], [30, 42.0], [31, 2.0], [32, 3.0], [33, 1.0], [34, 0.0], [35, 42.0], [36, 12.0], [37, 2.0], [38, 0.0], [39, 0.0], [40, 0.0], [41, 0.0], [42, 49.0], [43, 16.0], [44, 4.0], [45, 0.0], [46, 0.0], [47, 0.0], [48, 0.0], [49, 0.0], [50, 55.0], [51, 23.0], [52, 4.0], [53, 1.0], [54, 1.0], [55, 0.0], [56, 0.0], [57, 0.0], [58, 0.0], [59, 306.0], [60, 144.0], [61, 36.0], [62, 13.0], [63, 2.0], [64, 0.0], [65, 0.0], [66, 0.0], [67, 0.0], [68, 0.0], [69, 1530.0], [70, 841.0], [71, 223.0], [72, 63.0], [73, 16.0], [74, 11.0], [75, 8.0], [76, 11.0], [77, 5.0], [78, 0.0], [79, 0.0], [80, 3502.0], [81, 1743.0], [82, 438.0], [83, 100.0], [84, 41.0], [85, 60.0], [86, 104.0], [87, 74.0], [88, 17.0], [89, 1.0], [90, 0.0], [91, 0.0], [92, 3204.0], [93, 1568.0], [94, 484.0], [95, 115.0], [96, 57.0], [97, 58.0], [98, 135.0], [99, 75.0], [100, 20.0], [101, 1.0], [102, 0.0], [103, 0.0], [104, 0.0]]
#22109.0

NN_1 = [[0, 1671.0], [1, 3258.0], [2, 1296.0], [3, 288.0], [4, 71.0], [5, 40.0], [6, 37.0], [7, 37.0], [8, 21.0], [9, 2.0], [10, 0.0], [11, 0.0], [12, 0.0], [13, 0.0], [14, 0.0], [15, 0.0], [16, 0.0], [17, 1.0], [18, 2.0], [19, 0.0], [20, 10.0], [21, 1.0], [22, 0.0], [23, 0.0], [24, 36.0], [25, 8.0], [26, 1.0], [27, 1.0], [28, 1.0], [29, 64.0], [30, 42.0], [31, 2.0], [32, 3.0], [33, 1.0], [34, 0.0], [35, 42.0], [36, 12.0], [37, 2.0], [38, 0.0], [39, 0.0], [40, 0.0], [41, 0.0], [42, 49.0], [43, 16.0], [44, 4.0], [45, 0.0], [46, 0.0], [47, 0.0], [48, 0.0], [49, 0.0], [50, 55.0], [51, 23.0], [52, 4.0], [53, 1.0], [54, 1.0], [55, 0.0], [56, 0.0], [57, 0.0], [58, 0.0], [59, 306.0], [60, 144.0], [61, 36.0], [62, 13.0], [63, 2.0], [64, 0.0], [65, 0.0], [66, 0.0], [67, 0.0], [68, 0.0], [69, 1530.0], [70, 841.0], [71, 223.0], [72, 63.0], [73, 16.0], [74, 11.0], [75, 8.0], [76, 11.0], [77, 5.0], [78, 0.0], [79, 0.0], [80, 3502.0], [81, 1743.0], [82, 438.0], [83, 100.0], [84, 41.0], [85, 60.0], [86, 104.0], [87, 74.0], [88, 17.0], [89, 1.0], [90, 0.0], [91, 0.0], [92, 3204.0], [93, 1568.0], [94, 484.0], [95, 115.0], [96, 57.0], [97, 58.0], [98, 135.0], [99, 75.0], [100, 20.0], [101, 1.0], [102, 0.0], [103, 0.0], [104, 0.0]]
#22108.0

print(NN_2[0][1])
#1671.0
#for Schleife kein Ergebnis, warum? 
for i in range(105):
	if NN_2[i][1] != NN_1[i][1]:
		print(i)
'''

