#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:47:43 2021

@author: kraemer

data_creator_002.py
"""

#
# Dieses Programm erzeugt Daten indem es diejenigen Daten lädt, die von data_creator_001.py
# erzeugt wurden. Diese werden dann superpositioniert. Damit erzielen wir eine hohe Varianz
# an möglichen Szenarien. Ein damit trainiertes NN Modell sollte deutlich robuster sein als
# mit den unsuperpositionierten daten.
# Die erzeugten Daten beinhalten sEMG-Daten, die unterschiedliche Wellengeschwindigkeiten und Wellenlängen
# zur selben Zeit beinhalten. Der Speicherbedarf erhöht sich durch dieses Vorgehen.
# Da es einerseits interessant ist, die Auswirkung dieses Schritts zu messen und andererseits
# auch von Vorteil sein könnte, den zusätzlichen Speicherbedarf zu steuern, kann das mithilfe 
# des dritten Arguments getan werden.
# 
'''
 ------------------------------------------------------------------------------
 Im Folgenden steht viel Text, der erstmal gebraucht wurde, um dieses Programm
 zu entwickeln. Das sollte irgendwann ausgemistet werden, aber da es vielleicht
 auch zum Code-Verständnis beiträgt lasse ich es alles mal drin ...
 ------------------------------------------------------------------------------
 ------------------------------------------------------------------------------

 weil der Speicherbedarf relativ leicht explodiert, sollten wir eine Strategie dagegen halten.
===

 wir brauchen weiter auch Fälle, die "rein" sind.
   einen Teil beiseite legen?
   --> es geht erstmal Potenzial verloren. Macht Vergleich der Versionen 001 
       und 002 schwieriger, bzw. unmöglich.
   oder einfach zusätzlich welche erzeugen?
   --> der Bedarf an Speicherplatz ist nur marginal höher, wenn alles reine
       ganz verwendet wird. Man hat also nur einen Zugewinn an Potenzial.
       Vergleich von Versionen 001 und 002 direkt möglich.

 Wir nutzen also die 001 Daten mit und Superpositionieren diese dann geschickt.
-----

wir wollen
+ reine daten
+ Daten mit unterschiedlichen Wellenlängen
+ Daten mit unterschiedlichen Geschwindigkeiten (möglichst von allen Klassen!!!)
+ dabei sollen die Klassen keine unrealisitischen Szenarien beinhalten.

Vorgehen
-------
man könnte die einzelnen Kombinationen abzählen und erzeugen. Das ist aber nachteilig,
da die Anzahl der Klassen erst im Nachhinein bestimmt wird. Man kann nicht auf alle vor-
bereitet sein, weil es quasi unbegrenzt viele Möglichkeiten gibt.

Man kann aber versuchen, möglichst viel abzudecken, eine Metrik bereitstellen, welche
die Grenzen klar quantifiziert und nach dem Trainig schauen, ob das Modell dadurch limi-
tiert ist, oder es die fehlenden Daten "extrapolieren" bzw. sich erschließen konnte.

Wie decken wir möglichst viel ab?

Ansatz: Gibt es eine maximale anzahl an wellen?
(Überlegung: 2 mal die selbe welle führt zum gleichen ergebnis, aber die metadaten verdop-
peln sich) wir sollten uns vielleicht auf die Fälle beschränken, in denen der Mensch sich
noch zurecht findet: Zig übereinanderliegende Wellen (5 von 10 Arten) sind für kein Auge
mehr unterscheidbar/identifizierbar. Warum soll das ein NN können? Besser wir erhöhen das
Maximum nicht und belassen es bei dem bisher existierenden!
 1. Wir finden das Maximum an vorhandenen Wellen. Diese definieren die 100%-Marke.
 2. Alles was 100% erreicht, wird nicht mehr superpositioniert.
 3. wir sorgen dafür, dass die Fälle zwischen ~0% und 100% vorhanden sind und zwar nicht
    nur als "reine" Fälle. Wie?
      - alle kombis anschauen?
        (kombis begrenzt. wenn z.B. max 5 Wellen, aber 10 Geschwindigkeiten, dann kann
         nicht jede Geschw. mit drin sein. aber es können mehrere Formen auftreten. top?)
      - randomisieren? (wenn, dann mit random seed)
   - können die einzelnen Fälle als mini szenario angelegt werden? - jetzt ja. __getitem in meta und scenario ergänzt.
   - es ist zu beachten, dass wir noch nicht die Größe der zu verwendenden Bilder kennen!
     DAS HEIßT, WIR MÜSSEN DIE EINZELNEN FÄLLE MÖGLICHST GROß HALTEN!
     
     --> erstellen, auf max testen und gegebenenfalls Teile verwerfen? Das könnte relativ gut gehen.
         +: man kann es solange machen bis man alle Fälle (nicht alle Klassen) hat (Speicher?).
            - auf Klassen überprüfen kann man später.
            
Zur Überlegung "2 mal die selbe welle führt zum gleichen Ergebnis, aber die metadaten verdop-
peln sich": die Varianz an Anzahl gleicher Geschwindigkeiten und gleicher Wellenlänge haben
wir doch schon in den Grunddaten! Das heißt: WIR MISCHEN NICHT NOCHMAL DIESE!
  !GROßES +! ::  bei Betrachtung von genug Zeitschritten bleibt die Unterscheidungsfähigkeit
                 selbst mit bloßem Auge erhalten!
  kl.Nachteil :  gleiche geschwindigkeit, aber unterschiedliche Wellenlänge tritt nie in
                 einem Bild auf. (ist aber auch nicht in den Testdaten drin.. )
                 
     Problem im Vorgehen: die wellenlänge ist nicht mehr aufzufinden, wenn die Daten einmal
                          gemischt sind. Nur Superposition gleicher Geschwindigkeiten kann ver-
                          hindert werden.
                       +: wenn so akzeptiert (l=l aber v!=v), tritt obiger kl.Nachteil nicht auf.
                       -: es könnte zu schwer oder gar nicht erkennbaren Überlagerungen kommen,
                          (Das Problem ist aber nicht so stark wie bei v=v UND l=l)
'''

#import scenario as sc
#import data_creative_tools as dct
from update_dataFile import convert_data, convert_file
import numpy as np
import pickle
import sys
import os
import errno
import random

###############################################################################
def read_cla():
  input_list=[0,0,4,True]
  if len(sys.argv)  < 2:
    raise Warning("Please specify the number of speeds and wavelengths. (Data sets must already exist in the DATA/creator_001/ directory.)")
  input_list[0]= int(sys.argv[1])
  # empfohlene Werte n in [2,30]
  input_list[1]= int(sys.argv[2])
  # empfohlene Werte n in [10,30]
  if len(sys.argv) > 3:
    input_list[2]= int(sys.argv[3])
  # default behaviour: wir führen das ein, da es sehrwohl oft einen Unterschied macht, in der Betrachtung, wie viele Wellen maximal im System sind!
  '''
  Folgende Information kann getriggert werden, wenn in der Methode Scenario.maxWaves das dritte Argument auf True gesetzt wird (Arg 2 und 3 sind optionale Argumente)
  Warning for these scenarios, the probabilistic and integer point of view on the maximum amount of concurrent waves differ strongly: 6 vs 3
  Warning for these scenarios, the probabilistic and integer point of view on the maximum amount of concurrent waves differ strongly: 7 vs 4
  Warning for these scenarios, the probabilistic and integer point of view on the maximum amount of concurrent waves differ strongly: 4 vs 1
  Warning for these scenarios, the probabilistic and integer point of view on the maximum amount of concurrent waves differ strongly: 9 vs 6
  
  An dieser Stelle führen wir folgende Unterscheidung ein:
    Die Wellenanzahl wird
    a) großzügig NACH OBEN abgeschätzt: 'sobald ein minimales Signal der Welle in der Domain ist, wird sie mitgezählt.'
       Das ist 'count_waves_probabilistic'.
       a) ist DEFAULT Fall
    b) HART gesetzt, anhand von "mindestens x Prozent einer Welle sind in der Domain" - in diesem Fall gibt es für x auch einen default-Wert, der bei 0,5 liegt und bereits hart in den Daten ist. '(siehe creator_001).
       Das ist NICHT 'integer_interpretation' (was in die Modellbildung gehört)
               SONDERN 'not count_waves_probabilistic'
       führt zu Dateipfad ../count_waves_hard/..
       
  Default ist b) : count_waves_probabilistic = True
  '''
  if len(sys.argv)  > 4:
    input_list[3]= bool(int(sys.argv[4])) # from string to int to bool. (This needs int()-conversion inbetween!!)
    if input_list[3]:
      print("Warning: you entered in command line argument 4 but did not change the default behaviour (being 'count_waves_probabilistic = True').")
  return input_list
###############################################################################
# default settings and initialization:
n_speeds,n_waveLengths,grow_factor,count_waves_probabilistic = read_cla()
#print([n_speeds,n_waveLengths,grow_factor,count_waves_probabilistic])

# recreate name of the file with the base data // same as in data_creator_001.py 
file_name = "DATA/creator_001/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths.pickle"
  
# load it
convert_file(file_name)

with open(file_name, "rb") as f:
  daten = pickle.load(f)
scenarios = np.array(daten["content"])

# 1.: getmax number of waves in data set:
if isinstance(daten["maxW"],tuple):
  print('Neue Daten. maxW = {}.'.format(daten["maxW"]),end=' ')
  '''
  An dieser Stelle wird die Datenerzeugung geteilt, motiviert durch die unter-
  schiedliche Interpretation "ob Welle oder nicht". Die neuere (zusätzliche)
  Betrachtungsweise ist in einem extra Ordner zu finden (*002/integer_interp*).
  Die Benutzung der 'andersartigen' Daten im jeweils anderen Fall, ist aber
  nicht ausgeschlossen!
  Man könnte zum Beispiel überprüfen, ob die Performanz eines Modells für beide
  Trainingsdatensätze gleich ist.
  '''
  # newer representation of max Waves determined by: (first-integer) / (second-probabilistic) representation
  global_max = daten["maxW"][int(count_waves_probabilistic)]
  print("Choosing {}.".format(global_max))
else:
  global_max = daten["maxW"]
  print("Alte Daten. maxW = {} (Entspricht 'count waves probabilistic').".format(global_max))
#global_max = np.array(daten["maxW"]) # daten["maxW"] is a tuple (global max integer, global max probabilistic)


# determine a degree of accuracy that makes sense, based on the given maximum
# of waves in the system:
  # if we have max 1 wave, only 0 or 100% make sense. there is no situation of having 50 % of waves.
  # if we have max 2 waves, only 0, 50 0r 100% may occurr.
  # n waves --> 0, 1/n*100, 2/n*100, ... (n-1)/n*100, or 100%
  # Thus, the granularity that can be found in the scenarios is prescribed by the data.
  # on one hand, we have to make sure, that this information is contained in the training data,
  # on the other hand, we can be satisfied with that. there is no reason to put 
  # in more effort than this.
  #
  # one more thought on the granularity: it should be 100/m, with m \in N. that's implicit the case.
  #
  # m being global_max+1 yields the situations [i*100/global_max for i in range(global_max+1)]
  
 ## second thought on this issue:
   # if one considers the existence of a wave in a probabilistic way, for example if one is unsure, whether a wave is already 
   # inside the domain, or not, then the existence of a wave is in no way binary anymore. meening, that we can definitively
   # have nongranular (floating) representation of that alternative truth!
   # What does this imply w.r.t. this programme?
   #   1. The following two variables were defined but never used: 
   #     percentages_to_be_represented = [i*100/global_max for i in range(global_max+1)]
   #     wavenumbers_to_be_represented = [i for i in range(global_max+1)]
   #   conclusion: the previous thought were probably a dead end. There might
   #               be nothing left to be considered.

# the amount of storage for the new ( = original + additional) data set shall be bound.
# for the sake of simplicity, we use a grow factor (grow_factor ~= new/original) and
# make sure, that this factor is a lower bound like:
#   while actual_factor < grow_factor:
#     data_set.increase()
#     actual_factor.actualize(data_set)


'''wir können zum besseren Mischen die scenarios (typischerweise Längen von 600-2000 Zeitschritten (ts))
 in gleich große chunks (etwa 100ts) zerteilen, durchmischen und diese dann superpositionieren.
 
 das machen wir jetzt einfach mal, danach schauen wir, welche Teile den global_max-Test bestehen
 und wie groß diese Teile dann noch sind. sind sie größer 10, können sie schon relativ gut benutzt werden. größer 30, dann ist super..
 --> FAZIT nach Implementierung: dieses Vorgehen funktioniert für kleinere Datensätze sehr gut. grow_factor 4 wird dort gut erreicht.
     Für größere Daten (z.B. n_speeds=n_wavelengths=20 oder höher, kann es doch recht lange dauern bis man fertig ist)

  mit random.Random(fix_integer).shuffle(list) können wir die scenarien durchmischen

'''
def get_chunks_from_raw(chunksize,AofAs,Optional=False):
  '''
  this function only works for scenarios that are stored in an array-of-arrays (nxm) fashion
  AND that are at least (chunksize)ts long.
  
  returns broken (smaller) scenarios (chunks) of size chunksize within a shuffled list.
  we assume that this function is only used on raw data. Then, all chunks fulfill the global_max condition.
  
  if optional value given, we use this value to give back the total amount of time steps in the data set.
  Note that we immitate a pass-by-reference action here, since in python this concept does not exist.
  '''
  # the array to be created/returned:
  chunks = []
  
  speeds = len(AofAs)
  wavelens = len(AofAs[0])
  
  total_ts = 0
  
  for v in range(speeds):
    for l in range(wavelens):
      # get the length of this Scenario object:
      object_length = AofAs[v][l].data_.shape[0]
      total_ts += object_length
      # mark the start position of the considered timestep range:
      temp_pos = 0
      while temp_pos+chunksize <= object_length:
        # all complete chunks are turned into new Scenario objects:
        newChunk = AofAs[v][l][temp_pos:temp_pos+chunksize]
        chunks.append(newChunk)
        # increment the start position
        temp_pos += chunksize
      size_of_rest = object_length - temp_pos
      if size_of_rest > 1/3 * chunksize: # 1/3 could be anything ...
        # last portion of this scenario (AofAs[v][l][temp_pos:]) is used as well
        last_start_pos =  object_length - chunksize
        lastChunk = AofAs[v][l][last_start_pos:object_length]
        chunks.append(lastChunk)
      #else:
        # last portion of this scenario (AofAs[v][l][temp_pos:]) is tossed
  random.Random(0).shuffle(chunks)
  
  if Optional:
    Optional[0] = total_ts
  
  return chunks


'''
der einfachheit halber könnte man scenarios komplett wegwerfen, wenn sie den global_max-Test nicht bestehen.
wenn man das nicht tut, muss man bei allen übrigen Teilbereichen erstmal testen, wie lange sie denn sind.
Daher begnügen wir uns mal mit der einfacheren Version
'''

# DEPRECATED
def is_within_global_maxNOTEXISTING(scenario,gmax):
  '''
  scenario: the Scenario object to be tested
  gmax: the global maximum to be tested for

  this function carries out the global_max-test.
  A Scenario object passes the test if there is no time step in which the maximum of
  occurring waves exceeds the global_max of the raw data set.
  '''
  for ts in range(scenario.data_.shape[0]):
    for key in scenario.meta_.pData_[ts].keys():
      if max(scenario.meta_.pData_[ts][key].keys()) > gmax:
        #print(max(scenario.meta_.pData_[ts][key].keys()))
        #print(scenario.id_)
        return False
  return True

def get_speeds(scenario):
  '''
  scenario: the scenario that is checked for
  
  returns sorted list of contained speeds within the whole time range
  '''
  speeds=[]
  if isinstance(scenario,list): 
    for spds in [get_speeds(scenario[i]) for i in range(len(scenario))]:
      in_new = set(spds)
      in_speeds = set(speeds)
      in_second_but_not_in_speeds = in_new - in_speeds
      speeds = speeds + list(in_second_but_not_in_speeds)
  else:
    for ts in range(scenario.data_.shape[0]):
      for speed in scenario.meta_.iData_[ts].keys():
        if speed not in speeds:
          speeds.append(speed)
  return sorted(speeds)
  
def compatible(scenario1,scenario2):
  list_1 = get_speeds(scenario1)
  list_2 = get_speeds(scenario2)
  return not any(item in list_2 for item in list_1)

# approximate the number of new scenarios we need to produce and get the first set of chunks:

chunk_size = 100
total_timesteps = ['toBeOverridden']
base_chunks = get_chunks_from_raw(chunk_size,scenarios,total_timesteps) # <-- here, chunks are uniquely mixed
n_base = len(base_chunks)
print("base scenarios (first set of chunks):{}".format(n_base))
n_scenarios_to_be_produced = int(np.ceil(n_base*(grow_factor-1)))

# rename the list, since we will append all new ones herein:
all_scenarios=base_chunks

# nach vielen superpositionen kann es sein, dass keine validen Ergebnisse mehr gefunden
# werden. Das (und den maximal erreichten grow_factor) müsste man eventuell als Warnung
# bzw. als Info an den User zurück geben.
number_of_fails_in_a_row = 0
n_allowed_fails = 10

while len(all_scenarios) < n_scenarios_to_be_produced + n_base:
  # put them together in a clever way:
  # choose two "uniquely chosen" (random but reproducable)
  # using np.random.choice here, since random.choice does not allow for seeding!
  '''
  die momentane Vorgehensweise hat eventuell den Nachteil, dass ein Fokus der
  Scenarios auf wenig Superposition liegt. Je höher man aber grow_factor wählt
  desto mehr höher-wertige Superpositionen bekommt man.
  '''
  #np.random.seed(len(all_scenarios))
  # muss erst eingreifen um 
  random_scenario1 = np.random.choice(all_scenarios)
  random_scenario2 = np.random.choice(all_scenarios)
  random_tries=0
  # make sure they satisfy the "distinguishabilty-condition" -- Heißt: wenn ich viele szenarien einer geschwindigkeiten übereinander lege, dann kann man selbst bei ewiger betrachtung nicht zählen wie viele wellen es sind. Davor bewahren wir uns.
  while not compatible(random_scenario1,random_scenario2):
    random_tries+=1
    # we have to search another one. we try to minimize the total amount of waves,
    # so we skip the one with the most waves.
    if random_scenario1.maxWaves(count_waves_probabilistic) > random_scenario2.maxWaves(count_waves_probabilistic):
      #print("skipping scenario 1 (with max={})".format(random_scenario1.maxWaves()))
      random_scenario1 = np.random.choice(all_scenarios)
    else:
      #print("skipping scenario 2 (with max={})".format(random_scenario2.maxWaves()))
      random_scenario2 = np.random.choice(all_scenarios)
  #print("computed "+str(random_tries)+" for nothing")
  superposition = random_scenario1 + random_scenario2
  # test them and if passing global_max-test append to new_scenarios
  #if is_within_global_max(superposition,global_max):
  if superposition.maxWaves(count_waves_probabilistic) <= global_max:
    all_scenarios.append(superposition)
    number_of_fails_in_a_row  = 0
    #print("..having {} ts\n".format(superposition.data_.shape[0]))
  else:
    number_of_fails_in_a_row += 1
    print("Having {}/{} scenarios".format(len(all_scenarios)-n_base,n_scenarios_to_be_produced))
    #print("Scenario failed test. maxima: {} and {} (global_max = {})".format(random_scenario1.maxWaves(),random_scenario2.maxWaves(),global_max))
    if number_of_fails_in_a_row > n_allowed_fails:
      print("\n\nWARNING: I'm struggling to find sensuable superpositions.")
      appr_gf = ((len(all_scenarios)-n_base)*chunk_size + total_timesteps[0])/total_timesteps[0]
      print("Stopped further enhancement with grow factor = {} instead of {}.\n\n".format(appr_gf,grow_factor))
      break

if number_of_fails_in_a_row <= n_allowed_fails:
  appr_gf = ((len(all_scenarios)-n_base)*chunk_size + total_timesteps[0])/total_timesteps[0]
  print("Completed task with grow factor = {} >= {}.\n\n".format(appr_gf,grow_factor))

velocities = get_speeds(all_scenarios) # comment: this holds also velocities = 0. überlegung, ob das mit den 0-occurrences sinn macht.. am 'Rand zumindest würde ich solche gerne wegschneiden..


#format1 = {'content_tmp': all_scenarios, 'velocities': velocities, 'maxW': global_max}
daten['content'] = all_scenarios

# get unique and descriptive name for the new file

if count_waves_probabilistic:
  file_name = "DATA/creator_002/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths___grow_factor_"+str(grow_factor)+".pickle"  # Case a) (Default)
else:
  file_name = "DATA/creator_002/count_waves_hard/RawData_"+str(n_speeds)+"_velocities___"+str(n_waveLengths)+"_waveLengths___grow_factor_"+str(grow_factor)+".pickle" # Case b)

print('writing file {}'.format(file_name))


if not os.path.exists(os.path.dirname(file_name)):
    try:
        os.makedirs(os.path.dirname(file_name))
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

# save it (deletes old file if this already existed:)
with open(file_name, "wb") as f:
  pickle.dump(daten, f)

