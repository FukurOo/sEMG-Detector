import numpy as np 
import raw_to_input as inp
# import tensorflow as tf  could be used to use alternative dataset object.
from tensorflow import keras
from tensorflow.keras import layers
import pickle

class NeuralNetwork:
  '''
  instead of throwing around with variables in the code, we introduce this class.
  all relevant variables are then just easily accessible via this one common place.
  
  Frage: Wenn ein Modell gelade n wird. Ist dann nicht auch interessant, welches Netzwerk das Ding zustande bekommen hat?
  Sprich: Ich schau mir doch ein Modell nur an, um zu sagen, ob das Netzwerk taugt, bzw. mehrere um zu sehen, was sich lohnt...
  Also speichere ich ein Modell ab, inklusive des Netzwerks. Bzw. anstatt ein Modell zu öffnen, kann ich auch sagen - ich öffne ein Trainiertes Netzwerk und das Modell ist Teil von dem Netzwerk.
  Damit habe ich zugriff auf alle einstellungen. So rum muss es glaub sein.
  '''
  
  def __init__(self,training_options):
    '''
    configure the base settings for this Neural Network according to the users
    command line arguments
    '''
    ###TODO: anpassen. ist noch dict. jetzt soll aber InputPairs
    if 'AR' in training_options.keys():
      self.architecture = training_options['AR'].getV()
    else: 
      raise ValueError("architecture not specified!")
    if 'EP' in training_options.keys():
      self.epochs = training_options['EP'].getV()
    else: 
      raise ValueError("epochs not specified!")
    if training_options['CL'].getV() in ['r','n']:
      self.regression = True
      self.outputDimensionSize = training_options['S'].getV()
    else:
      self.regression = False
      self.NSections = training_options['S'].getV()
      if training_options['CL'].getV() in ['g','m']:
        #self.manualMode = True
        self.autoMode   = False
        self.NBricks = training_options['B'].getV()
      elif training_options['CL'].getV() in ['d','a']:
        self.autoMode   = True
        self.NBricks = 0
        #self.manualMode = False
    self.classification = not self.regression
    if training_options['GT'].getV() == 'i':
      self.integerDataInterpretation = True
    else:
      self.integerDataInterpretation = False
    self.probabilisticDataInterpretation = not self.integerDataInterpretation
    self.inputDimension = [training_options['TS'].getV(),0] # set the first input dimension
    #
    raw_data = pickle.load(open(training_options['DT'].getV(), "rb")) ## a temporary file (read later again temporarily..)
    self.tmp_wavecountis_prob = ('count_waves_hard' in training_options['DT'].getV())
    def myGet_dim(mylist):
      dim=[]
      while type(mylist) == type([]):
        dim.append(len(mylist))
        mylist=mylist[0]
      return dim
    if type(raw_data) != dict:
    # set up standardized data format as dictionary
    # this concerns all scenario data files older than 18th Nov. '21.
    # files that are younger, MUST TAKE ON THE FOLLOWING FORM already!
      data_as_dict = {}
      data_as_dict['content'] = raw_data
      #data_as_dict['vmin'] = 1.0 # default this. 
      #data_as_dict['vmax'] = 10.0 ###TODO: default this. however, when committing, make sure, al data_creators behave as expected!
      data_as_dict['maxW'] = 0
      data_as_dict['velocities'] = False
      data_as_dict['dimension'] = myGet_dim(raw_data)
      raw_data = data_as_dict.copy()
      del data_as_dict
    elif 'content_tmp' in raw_data.keys():
      assert False, " Fall wird noch benötigt." # wenn das nie auftaucht, könnte man es gut löschen... =D stupid comment
      data_as_dict = {}
      data_as_dict['content'] = raw_data['content_tmp']
      data_as_dict['velocities'] = raw_data['velocities']
      data_as_dict['maxW'] = raw_data['maxW']
      try:
        data_as_dict['vmin'] = raw_data['vmin']
      except:
        data_as_dict['vmin'] = raw_data['velocities'][0]
      try:
        data_as_dict['vmax'] = raw_data['vmax']
      except:
        data_as_dict['vmax'] = raw_data['velocities'][-1]
      data_as_dict['dimension'] = myGet_dim(raw_data['content_tmp'])
      raw_data = data_as_dict.copy()
      del data_as_dict
    elif 'content' in raw_data.keys():
      raw_data['dimension'] = myGet_dim(raw_data['content'])
      raw_data['vmin'] = raw_data['velocities'][0]
      raw_data['vmax'] = raw_data['velocities'][-1]
      if 'maxW' in raw_data.keys():
        if isinstance(raw_data['maxW'],tuple):
          raw_data['maxW'] = raw_data['maxW'][1-int(self.integerDataInterpretation)]
        else:
          raw_data['maxW'] = raw_data['maxW']
      else:
        raw_data['maxW'] = 0
        print("Warning: Old Data version might be outdated.")
    self.raw_data = raw_data # delete this large object after creating complete training data !
    
  def __str__(self):
    ret = ''
    ret += "inputDimension: {}\n".format(self.inputDimension)
    try:
      ret += "numberOfClasses: {}\n".format(self.numberOfClasses)
    except:
      print()
    ret += "maxWaves: {}\n".format(self.maxWaves)
    ret += "nVelocities: {}\n".format(self.nVelocities)
    ret += "velocities: {}\n".format(self.velocities)
    return ret
    
  def saveMemorySpace(self,radical=False):
    print("DELETING READ IN RAW DATA.")
    del self.raw_data
    if radical:
      del self.trnData
      del self.model
      del self.history
  
  def set_picShape(self,number):
    self.inputDimension[1] = number # set the second input dimension
  
  def set_trainingData(self,data):
    'test for the right structure here?'
    self.trnData = data
    
  def set_validationData(self,data):
    """
      data: this is the shuffled data. So NOT the original.
    """
    'test for the right structure here?'
    self.valData = data
    
  def set_traceback(self,IDs,g_truth=None):
    """
       Store information to trace the validation data back to
       the original image
       IDs: numpy list of integers.
            the first entry contains the index of the original position.
   g_truth: list of dicts
            Every entry is a dictionary, containing the original meta information for frames [0,1,...].
            IDs=[7,...] means: self.valData[0] is related to g_truth[7]
    """
    self.gtruth_id = IDs
    self.gtruth = g_truth
    
  def set_numberOfClasses(self,number):
    if not self.classification:
      print("Warning! Number of Classes only makes sense in a classification task setting!") # wie ist das mit SMap und maxWaves?
    else:
      self.numberOfClasses = number
      
  def set_sceneMapping(self,mapping):# this map is needed for classification tasks. it relates a class name to the physical meaning of it.
    self.SMap = mapping
  
  def set_maxWaves(self,N):###TODO: soll in file der rohen Scenario data stehen. da raus lesen. nicht selbst berechnen.
    if self.raw_data['maxW'] != N and self.raw_data['maxW'] != 0:
      assert False, "Determination of maxWaves odd! {} neq {}!".format(self.raw_data['maxW'], N)
      print("Warning: Determination of maximum number of waves seems to be odd! {} neq {}.".format(self.raw_data['maxW'], N))
    self.maxWaves = N
    
  def set_velocities(self,lst):
    self.velocities = lst
    self.nVelocities = len(lst)
    assert (self.raw_data['vmin'] == lst[0] and self.raw_data['vmax'] == lst[-1]),"Warning:Determination of minimum / maximum velocities seem to be odd! {} neq {} / {} neq {}.".format(self.raw_data['vmin'], lst[0], self.raw_data['vmax'], lst[-1])
    self.vmin = lst[0]
    self.vmax = lst[-1]
  
  def getTrainingData(self): # könnte auch setup() heißen
    '''
    read raw EMG data. need to transform it w.r.t. some kind of standard
    
    the standard might (but should preferably not) depend on which NN API is used (e.g. keras or pytorch)
    '''

  # settings that are independent of the kind of task:
    #print(type(self.raw_data['content']))
    #print(self.raw_data['dimension'])
    print("In den Daten enthaltene Geschwindigkeiten: {}".format(self.raw_data['velocities']))
    self.set_velocities(self.raw_data['velocities'])
    print("In den Daten enthaltene Maximalwellenzahl: {}".format(self.raw_data['maxW']))
    data_array = []
    if len(self.raw_data['dimension']) == 1:
      data_array = self.raw_data['content']
    elif len(self.raw_data['dimension']) == 2:
      for lst in self.raw_data['content']:
        for el in lst:
          #theoretically one could also check here, whether the element is a Scenario object.
          data_array.append(el)
    else:
      raise Exception("Unknown data format in given file!")
    #self.saveMemorySpace()
    self.raw_data['content'] = data_array # instead of making a copy and deleting the old thing (including the other keys!), we override the old content key by a link to the uniform data_array. so all other keys still are available

    self.set_picShape(data_array[0].number_of_electroids_)
    total_frame_number = 0 # the number of 'pictures' that the image detection is trained with
    for num in range(len(data_array)):
      scenario = data_array[num]
      total_frame_number += scenario.meta_.size_ - (self.inputDimension[0]-1)
        
    x_temp = np.ndarray([total_frame_number,self.inputDimension[0],self.inputDimension[1],1]) # this is the input stream
    
    # set up a pre version of the output stream. it needs to be processed, depending on the kind of task that is to be used.
    groundtruths=[]
    filled_elements = 0
    for num in range(len(data_array)):
      grayscale_pictures,groundTruth = inp.extractFrames(self.inputDimension,data_array[num],self.integerDataInterpretation) # after this, groundTruth holds the raw dictionaries (states) that still need to be related to a class number / class name (done so in classify())
      end = filled_elements + grayscale_pictures.shape[0]
      x_temp[filled_elements:end] = grayscale_pictures
      filled_elements = end
      for el in groundTruth:
        groundtruths.append(el)
   
  # settings that are dependent on the kind of task:
    #self.set_velocities(self.raw_data['velocities'])
    if self.classification: 
      import classification_task as clat
      # so lange die creators noch nicht die neue maxWaves berechnung aus basics benutzen, muss im folgenden statement maxW = 0 (letztes Argument) stehen!
      y_temp,maxWaves,number_of_classes,Map = clat.classify(groundtruths,self.NBricks,self.NSections,self.integerDataInterpretation,self.velocities,self.tmp_wavecountis_prob,self.raw_data['maxW'])
      if self.autoMode:
        self.NBricks = maxWaves
      self.set_numberOfClasses(number_of_classes)
      self.set_sceneMapping(Map)
    elif self.regression:
      import regression_task as regt
      # so lange die creators noch nicht die neue maxWaves berechnung aus basics benutzen, muss im folgenden statement maxW = 0 (letztes Argument) stehen!
      maxWaves,y_temp = regt.get_output(groundtruths,self.integerDataInterpretation,self.outputDimensionSize,self.velocities,self.tmp_wavecountis_prob,self.raw_data['maxW'])# y_temp soll dimension haben: [Anzahl an Auswertungen, Anzahl an Regressions-Dimensionen (=self.outputDimensionSize)]
    self.set_maxWaves(maxWaves)
      
    # keep track of mappings as well. so we have to shuffle the IDs the same way as the training data
    velocity_Ids_sorted = [i for i in range(len(groundtruths))]
    velocity_Ids_sorted = np.array(velocity_Ids_sorted)
    velocity_Ids_sorted = velocity_Ids_sorted.reshape(len(velocity_Ids_sorted),1)
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
    velIds_train = np.stack(velocity_Ids_sorted[::5])
    velIds_train = np.vstack((velIds_train,velocity_Ids_sorted[1::5]))
    velIds_train = np.vstack((velIds_train,velocity_Ids_sorted[2::5]))
    velIds_train = np.vstack((velIds_train,velocity_Ids_sorted[3::5]))
    velIds_validation = velocity_Ids_sorted[4::5].copy()
    
    # something went wrong... some have shape=( ... ,1)  with unnecessary last dimension  ##NOT SO SURE ABOUT THIS! THIS DIMENSION MIGHT BE NECESSARY FOR KERAS!
    # for now, kill this dimension..
    #ToDo: eliminate reason and get rid of this reshaping. What are the cases, this occurs, and when not? This is very ugly here.
    for a in [x_train,x_validation]:#,velIds_train,velIds_validation]:  not working for the last two..
      shrinkToSize = len(a.shape)-1
      a = a.reshape([a.shape[i] for i in range(shrinkToSize)])
    
    if not (x_train.shape[0]==y_train.shape[0]==velIds_train.shape[0] and x_validation.shape[0]==y_validation.shape[0]==velIds_validation.shape[0]):
      print("\n\n WARNING!! original waves and training data may not coincide ! !")
      
    self.set_trainingData((x_train,y_train))
    self.set_validationData((x_validation,y_validation))
    if self.classification: 
      self.set_traceback(velIds_validation,groundtruths)
    else:
      self.set_traceback(velIds_validation)
    ###TODO; maxWaves soll eigentlich vorher schon existieren, aber vielleicht sollte ich zunächst ermöglichen, dass es auch mit den datensätzen funktioniert, die das nicht kennen..
    ##  (Map,velIds_train,velIds_validation,groundtruths) makes the original waves still available for investigations
    #return (velIds_train,velIds_validation,groundtruths)

  
  def startTraining(self):
    '''
 about: NORMALIZATION
 from https://www.tensorflow.org/tutorials/keras/regression#linear_regression_with_multiple_inputs
 
 It is good practice to normalize features that use different scales and ranges. 
 One reason this is important is because the features are multiplied by the model weights. So, the scale of the outputs and the scale of the gradients are affected by the scale of the inputs.
 Although a model might converge without feature normalization, normalization makes training much more stable.
 
 Da unsere input-Daten alle normalisiert sind (grauwerte zwischen 0 und 1), benötigen wir diesen Schritt nicht. 
 Man könnte ihn aber trotzdem einbauen, da man dann auch andere Daten verarbeiten kann (zb Datenformate 0 bis 255 oder floats > 1 ...)
 
'''
    inputs = keras.Input(shape=(self.inputDimension[0],self.inputDimension[1],1))
### TODO: aufräumen. Struktur für Steuerung von Klassifizierung oder Regression ?!
# # # welche struktur brauch ich? ist hier überhaupt noch ein Unterschied oder ist das jetzt schon vorbei?
###       1) Ordnung schaffen,  2) Regressionsmodelle entwerfen.
    x=inputs
    outputs = None
    if self.architecture == 0:
      x = layers.Conv2D(filters=32, kernel_size=(3, 3),padding='valid', activation="relu")(x)
      x = layers.MaxPooling2D(pool_size=(3, 3))(x)
      x = layers.GlobalAveragePooling2D()(x)
      outputs = layers.Dense(self.numberOfClasses, activation='softmax')(x)
      model = keras.Model(inputs=inputs, outputs=outputs)
    elif self.architecture == 1:
      x = layers.Conv2D(filters=16, kernel_size=(2, 2),padding='valid', activation="relu")(x)
      x = layers.MaxPooling2D(pool_size=(3, 3))(x)
      x = layers.GlobalAveragePooling2D()(x)
      outputs = layers.Dense(self.numberOfClasses, activation='softmax')(x)
      model = keras.Model(inputs=inputs, outputs=outputs)
    elif self.architecture == 2:
      conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3),padding='valid',activation=None)(x) # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/ uses ReLU activation, but we already have values larger 0!
    
      conv2 = layers.Conv2D(filters=32, kernel_size=(1, 3),padding='valid',activation=None)(conv1)
      conv2 = keras.layers.BatchNormalization()(conv2)
      conv2 = keras.layers.ReLU()(conv2)
    
      conv3 = layers.Conv2D(filters=32, kernel_size=(1, 3),padding='valid',activation=None)(conv2)
      conv3 = keras.layers.BatchNormalization()(conv3)
      conv3 = keras.layers.ReLU()(conv3)
    
      gap = keras.layers.GlobalAveragePooling2D()(conv3)
    
      outputs = layers.Dense(self.numberOfClasses, activation='softmax')(gap)
      model = keras.Model(inputs=inputs, outputs=outputs)
    elif self.architecture == 3:
      # first Regression model. based on keras.Sequential() (0,1,2 basically could be that as well..)
      # ALLGEMEIN BEACHTEN: Wir brauchen dense layers! diese können die Existenz mehrerer Wellen über die ganze Domain (also summieren) beachten! ein Kernel (3x3 o.ä.) kann das nicht!
      model = keras.Sequential()
      model.add(inputs)
      model.add(layers.Conv2D(filters=32, kernel_size=(3, 3),padding='valid', activation="relu"))
      model.add(keras.layers.Dense(max(16,self.outputDimensionSize*2)))
      print("model.output_shape is currently {}, and should be (None, {}).\n".format(model.output_shape,self.outputDimensionSize))
      model.add(layers.Flatten())
      print("model.output_shape is currently {}, and should be (None, {}).\n".format(model.output_shape,self.outputDimensionSize))
      model.add(layers.Dense(self.outputDimensionSize, activation="relu"))
      print("model.output_shape is currently {}, and should be (None, {}).\n".format(model.output_shape,self.outputDimensionSize))
    elif self.architecture == 4:
      # idee hier: 1 filter für jede Geschwindigkeit
      numberOfVelocities = len(self.raw_data['velocities'])
      model = keras.Sequential()
      model.add(inputs)
      model.add(layers.Conv2D(filters=numberOfVelocities, kernel_size=(3, 3),padding='valid', activation="relu"))
      model.add(keras.layers.Dense(max(int(numberOfVelocities/2),self.outputDimensionSize*2)))
      model.add(layers.Flatten())
      model.add(keras.layers.Dense(self.outputDimensionSize, activation="relu"))
###
    model.summary()
    if self.classification:
      ########## MÖGLICHE VARIATION: optimizers SGD, Adam, Adadelta, Adagrad, 
      ########## MÖGLICHE VARIATION: learning_rate
      #optim = keras.optimizers.RMSprop(learning_rate=1e-3)
      optim = keras.optimizers.Adadelta(learning_rate=1e-3)
  
      ########## MÖGLICHE VARIATION: keras.losses.CategoricalCrossentropy()
      lss = keras.losses.CategoricalCrossentropy() #keras.losses.MeanSquaredError()
 
    ########## MÖGLICHE VARIATION: metrics?
      if False:
        model.compile(optimizer=optim,loss=lss, metrics = ['accuracy'])
      else:
        model.compile(optimizer=optim,loss='mse', metrics = [keras.metrics.CategoricalAccuracy()])# loss = lss, metrics=['accuracy'])
    else:
      model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))#(loss='mae', optimizer='adam')
  # compute NN and store the progress for eventual use, later.
  ########## MÖGLICHE VARIATION: batch_size
    self.history = model.fit(self.trnData[0],self.trnData[1],batch_size=1024,epochs=self.epochs, validation_data = self.valData)
    self.model = model
  #end of function


  def saveModel(self,relative_file_path):
    import pickle
    import matplotlib.pyplot as plt
    from tensorflow.keras import utils
    import os
  
    # plot loss and accuracy, independent on choice
    loss_key=False
    val_loss_key=False
    acc_key=False
    val_acc_key=False
    
    for key in self.history.history.keys():
      if 'accuracy' in key:
        if 'val' in key:
          val_acc_key = key
        else:
          acc_key = key
      elif 'loss' in key:
        if 'val' in key:
          val_loss_key = key
        else:
          loss_key = key
      else:
        print("WARNING: Unidentified key in network history.")
    
    if acc_key and val_acc_key:
      # base settings for figure size:
      from matplotlib.pyplot import figure
      figure(figsize=(6,4), dpi=80)
      # summarize history for loss:
      training_steps =self.history.history[acc_key] # shorter name
      validation_steps=self.history.history[val_acc_key] # shorter name
      plt.plot(training_steps,linewidth=3)
      plt.plot(validation_steps,linewidth=1.8)
      # save this data if needed again:
      with open(relative_file_path+"/visualized_data.pkl", "wb") as f:
        pickle.dump(((training_steps,validation_steps)), f)
      # set all texts and labels:
      plt.title('model accuracy',fontsize=28)
      plt.ylabel(acc_key,fontsize=20)
      plt.xlabel('epoch',fontsize=20)
      plt.legend(labels=['train', 'test'], loc='upper left',fontsize=14)
      plt.xticks(fontsize=12)# wuwu
      plt.yticks(fontsize=12)# wuwu
      plt.savefig(relative_file_path+"/Accuracy")
      plt.savefig(relative_file_path+"/Accuracy.eps")
      plt.clf()
      # inform console about final result
      meean = sum(validation_steps[-3:])/3
      print(meean)
      with open(relative_file_path+"/reached_accuracy.txt", "x") as f:
        f.write(str(meean))
      
    if loss_key and val_loss_key:
      # base settings for figure size
      from matplotlib.pyplot import figure
      figure(figsize=(6,4), dpi=80) #12,7 auf poster, aber nachjustiert..
      # summarize history for loss:
      training_steps =self.history.history[loss_key] # shorter name
      validation_steps=self.history.history[val_loss_key] # shorter name
      plt.plot(training_steps,linewidth=3)
      plt.plot(validation_steps,linewidth=1.8)
      # save this data if needed again:
      with open(relative_file_path+"/visualized_data.pkl", "wb") as f:
        pickle.dump(((training_steps,validation_steps)), f)
      # set all texts and labels:
      plt.title('model loss',fontsize=28)
      plt.ylabel(loss_key,fontsize=20)
      #plt.ylabel('mean absolute error',fontsize=40)
      plt.xlabel('epoch',fontsize=20)
      plt.legend(fontsize=14,labels=['training', 'validation'], loc='upper right') # was left
      plt.xticks(fontsize=12)# wuwu
      plt.yticks(fontsize=12)# wuwu
      plt.savefig(relative_file_path+"/Loss")
      plt.savefig(relative_file_path+"/Loss.eps")
      plt.show()
      plt.clf()
      # inform user through console about final result and save in file:
      meean = sum(self.history.history[val_loss_key][-3:])/3
      print(meean)
      with open(relative_file_path+"/reached_loss.txt", "x") as f:
        f.write(str(meean))
  
    self.model.save(relative_file_path+"/model")
    utils.plot_model(self.model, to_file=(relative_file_path+"/model.png"))
  
    #if self.classification:
    #  scene_file_name = relative_file_path+"/scene_mapping.pickle"
    #  pickle.dump((self.SMap,self.valData), open(scene_file_name, "wb"))
    #else:
    #  pickle.dump(self.valData, open(scene_file_name, "wb"))
    #with open(scene_file_name, 'wb') as pfile:
    #  pickle.dump(self, pfile,protocol=pickle.HIGHEST_PROTOCOL)
  
  
    print("\n\n\n\n\n\n\n")
    print("Finishing up. Proceede by executing '. ./execute-me.sh <outfile>'.")
    command = "mv $1 "+relative_file_path+"/."+" && echo '... proceeding execution ...' && echo 'Done.'"
    write_c_to_file = "echo '"+command+"' > execute-me.sh"
    print()
    os.system(write_c_to_file)
    os.system("sleep 0.02")