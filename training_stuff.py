import numpy as np 
import scipy as sp
from scipy import special
# import tensorflow as tf  could be used to use alternative dataset object.
from tensorflow import keras
from tensorflow.keras import layers

def getNumberOfClasses(maxW,training_opts):
  #
  # ob wir integer oder probabilistic haben ist erstmal egal.
  # entscheidend ist allein:
  #  ob Geschwindigkeiten unterschieden werden -- training_opts['S']
  #  und ob wir eine directe oder granulare Betrachtung vornehmen. -- training_opts['CL']
  #  (bei der direkten formulierung ist die frage, ob die cross entropie tauglich ist oder nicht)
  #
  if (training_opts['CL']=='d'):
    # S>0: NN unterscheidet Klassen nach Geschwindigkeits-Gruppe und bemisst diese anteilig
    if training_opts['S']:
      return training_opts['S']
    # S=0: NN unterscheidet Klassen nach Anzahl an Wellen
    else:
      return maxW + 1
  elif (training_opts['CL']=='g'):
  #
  #  |_maxW               ____
  #  |                   |____|
  #  |          ____     |____|
  #  |         |____|    |____|
  #  |         |____|    |____|
  #  |     ____|____|____|____|
  #  |    |____|____|____|____|
  #  |____|____|____|____|____|______>
  #
  # Stone size in picture is maxW/7 so granularity, G is 100/7
  # or more consistent: G=100/7, so number of stones = 100/G = 7: 7 stones fill up to maxW
  # usually we can assume granularity to be G=1/n or G=n with n being integer and 100/G being integer as well  
    total_classes = 0
    number_of_stones = int(100/training_opts['G'])
    sections = training_opts['S']
    # sections must not be 0 in this case! Is made sure in argument_stuff.py
    for i in range(number_of_stones):
      total_classes += sp.special.comb(sections + number_of_stones - 1, number_of_stones, exact=True)
    if total_classes >= 500:
      print("\n\nWarning! This Neural Network is trained with a lot of classes! ({})\n\n".format(total_classes))
    return total_classes
    
def P_N_waves_are_there(b):
  k = [el for el in b.keys()]
  if len(k)==0:
    print("size error in function f")
  t_new={}
  for speed in b[k[0]].keys():
    t_new[speed] = b[k[0]][speed]
  #print(t_new)
  for i in range(1,len(k)):
    #for j in range(i):
    t_old = t_new.copy()
    t_new = {}
    for m in b[k[i]].keys():
      for n in t_old.keys():
        if (m+n) in t_new.keys():
          t_new[m+n] += b[k[i]][m]*t_old[n]
        else:
          t_new[m+n] = b[k[i]][m]*t_old[n]
    #print(t_new)
  return t_new

def classify(list_of_dicts,training_opts):
  # NoS: the Number of Sections (the spectrum of speeds is devided into) = training_opts['S']
  length = len(list_of_dicts)
  # for now this is not implemented very well ...
  # the number of waves is simply the class number
  count = np.ndarray([length],dtype=np.int32)
  maxW = 0
  for el in range(length):
    count[el] = 0
    for kel in list_of_dicts[el].keys():
      if (training_opts['GT']=='p'): # more complex data structure here:
        count[el] += int(max(list_of_dicts[el][kel].keys()))
      elif (training_opts['GT']=='i'):
        count[el] += int(list_of_dicts[el][kel])
    if count[el] > maxW:
      maxW = int(count[el])
      if type(maxW)!=type(2):
        print("typ von maxW Zeile 80 falsch")

  # calculate how many Classes there exist
  NoC = getNumberOfClasses(maxW,training_opts)
  
### Die metadaten enthält beide versionen, p und i zeitlgleich als metaObject.pData_ und metaObject.iData_
  output = np.zeros([length,NoC])
  for el in range(length):
    # integer:
    if training_opts['GT']=='i':
      output[el][count[el]] = 1 # hatte hier mal 'int(math.floor(count[el]+.5))' stehen
    # probabilistic:
    else:
      # hier wird es jetzt wieder komplizierter... wir müssen jede wahrscheinlichkeit einer Geschwindigkeit mit allen Möglichkeiten aler anderen Geschwindigkeiten multiplizieren.
      # bei zwei v geht das noch..:
      # {1.0:{0: 0.08, 1: 0.92}}  {2.0:{0: 0.8, 1: 0.2}}
      #   => [0.064, 0.736+0.016, 0.184]
      # bei dreien wird es schrott... aber wir können induktiv voranschreiten, bis keins mehr da ist..
      # Dazu können wir die dicts in eine Liste packen, die wir nach und nach wieder leeren.
      P_dict = P_N_waves_are_there(list_of_dicts[el])
      for N in range(maxW+1):
        if N > maxW:
          print("LOGIC ERROR in classify(), line 113, in toTrainingProb")
        if N in P_dict.keys():
          output[el][N] = P_dict[N]
        else:
          output[el][N] = 0
    
  # 'distribution' zeigt an, wie oft welche Anzahl an wellen auftaucht.
  # indiziert, ob das NN vielleicht fehleranfällig ist
  unique, counts = np.unique(count, return_counts=True)
  unq = [int(el) for el in unique]
  distribution = dict(zip(unq, counts))
  
  return output,maxW,distribution,NoC

def extractFrames(pic_size,scenario,training_opts,method=0):
  # this method returns
  #   * pictures (list of numpy arrays) and
  #   * the number of present waves for [each wave speed = key] for [each picture = index] (list of dictionaries).
  #
  # here, we need the kind of data interpretation,   GT = 'i' or 'p'
  #
  # 'height' simply defines how many time steps yield a picture.
  # the width of a picture correlates directly to the scenario
  # 'method' decides what strategy we use to set the "ground truth"
  # method==0 means, we take the most actual dictionary as a truth.
  # vermutlich werden wir keine andere Methode implementieren. es gibt genug anderes zu untersuchen..
  (height,width) = pic_size
  frame_list = np.ndarray([scenario.meta_.size_ - (height-1),height,width,1])
  raw_label_list = []
  if method == 0:
    print(end='')
  else:
    print("In extract_frames(): method not implemented yet.")
    method=0
  for stop in range(height,scenario.meta_.size_+1):
    first = stop - height
    last = stop - 1
    #print(str(first)+" "+str(last))
    pic = scenario.data_[first:stop]
    reshaped_pic = pic.reshape(1,pic.shape[0],pic.shape[1],1)
    frame_list[first] = reshaped_pic
    if (method == 0):
      if training_opts['GT']=='i':
        raw_label_list.append(scenario.meta_.iData_[last])
      elif training_opts['GT']=='p':
        raw_label_list.append(scenario.meta_.pData_[last])
  return frame_list,raw_label_list

def getTrainingData(data_array,training_opts):
  ####
  # hier werden benötigt:
  # "GT" (char): the kind of data interpretation. integer (i) or probabilistic (p)'
  # "CL" (char): the way classes are defined. granular (g) or direct (d)'
  # "S"   (int): the number of sections to be used'
  # "G" (float): the granularity if CL=g, otherwise meaningless(?!).'
  # "TS"  (int): the number of time steps that define a picture'
  ####
  # Nur TS ist dabei für die Bilder verantwortlich. Der Rest bezieht sich auf die Labels.
  # Für GT muss bereits beim Einlesen der Daten Acht gegeben werden: es gibt integer und probabilistic Daten!
  # .. eventuell muss man auch bei der Nachbehandlung aufpassen..
  image_size=(training_opts['TS'],data_array[0].number_of_electroids_) # (rows,columns)
  total_frame_number = 0
  for num in range(len(data_array)):
    total_frame_number += data_array[num].meta_.size_ - (image_size[0]-1)
  
  x_temp = np.ndarray([total_frame_number,image_size[0],image_size[1],1]) # this is the x_train

  groundtruths=[]  # this is something like the y_train
  filled_elements = 0
  for num in range(len(data_array)):
    grayscale_pictures,groundTruth = extractFrames(image_size,data_array[num],training_opts)
    end = filled_elements + grayscale_pictures.shape[0]
    x_temp[filled_elements:end] = grayscale_pictures
    filled_elements = end
    for el in groundTruth:
      groundtruths.append(el)
  
  y_temp,maxW,distribution,number_of_classes = classify(groundtruths,training_opts)
  #####################
  #### WE ALSO SHUFFLE AROUND THE DATA SETS A BIT, SO THE
  #### VALIDATION DATA IS REPRESENTATIVE AND NOT OF A NEW TYPE
  ##
  ## we use ~ 20 % of the data for validation:
  x_train = np.stack(x_temp[::5]) # DO NOT USE numpy vstack here!
  x_train = np.vstack((x_train,x_temp[1::5])) # USE vstack here
  x_train = np.vstack((x_train,x_temp[2::5]))
  x_train = np.vstack((x_train,x_temp[3::5]))
  x_validation = x_temp[4::5].copy()
  ##
  y_train = np.stack(y_temp[::5])
  y_train = np.vstack((y_train,y_temp[1::5]))
  y_train = np.vstack((y_train,y_temp[2::5]))
  y_train = np.vstack((y_train,y_temp[3::5]))
  y_validation = y_temp[4::5].copy()
  ##
  ##
  del x_temp
  del y_temp
  
  return image_size,(x_train,y_train),(x_validation,y_validation),number_of_classes

  
def trainNeuralNetwork(pic_shape,training_data,valdtn_data,num_classes,training_opts):
  inputs = keras.Input(shape=(pic_shape[0],pic_shape[1],1))
  
### TODO: hier bin ich. muss NN variieren.
  x=inputs
  outputs = None
  if training_opts['AR']==0:
    x = layers.Conv2D(filters=32, kernel_size=(3, 3),padding='valid', activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
  elif training_opts['AR']==1:
    x = layers.Conv2D(filters=16, kernel_size=(2, 2),padding='valid', activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
###
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.summary()
  
  ########## MÖGLICHE VARIATION: learning_rate
  optim = keras.optimizers.RMSprop(learning_rate=1e-3)
  
  ########## MÖGLICHE VARIATION: keras.losses.CategoricalCrossentropy()
  lss = keras.losses.MeanSquaredError()
  
  ########## MÖGLICHE VARIATION: metrics?
  model.compile(optimizer=optim,loss=lss, metrics=['accuracy'])
  
  # compute NN and store the progress for eventual use, later.
  ########## MÖGLICHE VARIATION: batch_size
  history = model.fit(training_data[0],training_data[1],batch_size=1024,epochs=training_opts['EP'], validation_data = valdtn_data)
  
  return history, model