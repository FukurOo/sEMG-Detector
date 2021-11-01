#from scenario import *
import pickle
import numpy as np

#f = open("DATA\creator_001\RawData_4_velocities___5_waveLengths.pickle",'rb')
#f = open("DATA\creator_001\RawData_3_velocities___20_waveLengths.pickle",'rb')
f = open("DATA\creator_002\RawData_3_velocities___20_waveLengths___grow_factor_4.pickle",'rb')
daten = pickle.load(f)

daten = np.array(daten)
#for d in daten :
#    print("1. Objekt")
#    print(d[0].length_of_adhesive_array_,"Länge des anhaftenden Arrays")
#    print(d[0].time_between_sEMG_samples_,"Zeit zwischen den sEMP Stichproben")
#    print(d[0].IED_,"Distanz zwischen den Elektroden")
#    print(d[0].manager_,"Manager")
    #print(d[0].meta_.get_vecs(),"meta")#
#    print(d[0].data_,"data")
#    print("2. Objekt")
#    print(d[1].length_of_adhesive_array_,"Länge des anhaftenden Arrays")
#    print(d[1].time_between_sEMG_samples_,"Zeit zwischen den sEMP Stichproben")
#    print(d[1].IED_,"Distanz zwischen den Elektroden")
#    print(d[1].manager_,"Manager")
    #print(d[1].meta_.get_vecs(),"meta")
#    print(d[1].data_,"data")
#    print("3. Objekt")
#    print(d[2].length_of_adhesive_array_,"Länge des anhaftenden Arrays")
#    print(d[2].time_between_sEMG_samples_,"Zeit zwischen den sEMP Stichproben")
#    print(d[2].IED_,"Distanz zwischen den Elektroden")
#    print(d[2].manager_,"Manager")
#    print(d[2].meta_.iData_,"meta_idata")
#    print(d[2].meta_.pData_,"meta_pdata")
#    print(d[2].meta_.size_,"meta_size")
#    print(d[2].data_,"data")
#    break

#[i.analyse() for i in daten]
print(daten)
print(np.shape(daten)[0])


for i in daten:
        i.analyse()
#    for j in range(np.shape(daten)[0]):
#        daten[i,j].show()
#        daten[i,j].analyse()
#        print('hallo du')


print("Lily")