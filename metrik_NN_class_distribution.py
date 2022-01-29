#dies ist die Metrik NN_class_distribution
#sie überprüft welche Klasse wie oft in einem Datensatz vertreten ist

import argument_stuff as USER
import file_stuff as MMRY
import training_stuff_2 as NN
import numpy as np
import matplotlib.pyplot as plt

#kopiert aus main.py
training_options = USER.processArguments()
data_file,output_path = MMRY.preProcess(training_options,'MODELS/')
pic_shape, training_data, validation_data, number_of_classes, scene_mapping  = NN.getTrainingData(data_file,training_options)


y_validation = validation_data[1]
#in reference_class werden die Klassen gesammelt. Dazu:
#y_validation[i] hat soviele Elemente wie es Klassen gibt
#für GL i steht überall eine 0, außer an der Stelle der richtigen Klasse.
#Dort steht eine 1. np.argmax gibt diese Stelle aus.
reference_class = [np.argmax(y_validation[i]) for i in range(0,y_validation.shape[0])]

#Histogramm
n, bins, patches = plt.hist(reference_class, bins = [i for i in range(number_of_classes+1)])
#Ausgabe der Häufigkeit aller Klassen im Terminal
print('[Klasse, Häufigkeit]')
print([[bins[i], n[i]] for i in range(len(bins)-1)])
print('Gesamt: 'np.sum(n))

plt.title("{}".format('Verteilung an NN-Klassen'))
plt.xlabel("Klasse")
plt.ylabel("Häufigkeit")
plt.show()
