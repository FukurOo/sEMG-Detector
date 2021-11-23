import numpy as np 
import scipy as sp
import matplotlib.pyplot as plt

from scipy import special
# import tensorflow as tf  could be used to use alternative dataset object.
from tensorflow import keras
from tensorflow.keras import layers

def getNumberOfClasses(numberOfBricks,numberOfSections):
  '''
  calculate and return the number of NN classes.

  Consider the following picture. There, the number of bricks is 7. The number
  of sections is 4. (sometimes we may use the words 'granularity' or 'brick size'
  which here is 100/7)
  Every possible state of how the graph could look like, is a class. In this
  graph, there are 5 bricks ('XXXX') out of 7 possible. This states name
  (or the name of the class) is '(0,3,2,0)'.
  
  Note: the number of bricks can be equal to the number of maximal waves.
  However, this is not necessarily the case!
  
  nOB_| ____ ____ ____ ____
      ||____|____|____|____|
      ||____|____|____|____|
      ||____|____|____|____|
      ||____|____|____|____|
      ||____|XXXX|____|____|
      ||____|XXXX|XXXX|____|
      ||____|XXXX|XXXX|____|__>
  
  We can calculate the number of classes explicitly. The number of NN classes is:   
  $$\sum_{n=0}^{N_B} \frac{(N_S + n-1)!}{n! (N_S - 1)!)} $$
  '''
  total_classes = 0
  for n in range(numberOfBricks+1):
    total_classes += sp.special.comb(numberOfSections + n - 1, n, exact=True)
    #print(sp.special.comb(numberOfSections + n - 1, n, exact=True))
  if total_classes >= 500:
    print("\n\nWarning! This Neural Network is trained with a lot of classes! ({})\n\n".format(total_classes))
  return total_classes
    
def getClasses(numberOfBricks,numberOfSections):
  '''
  Diese Funktion benennt alle Klassen, die es geben kann für die Kombination (nOB,nOS).
  Es sind natürlich genauso viele, wie in getNumberOfClasses berechnet.
  
  return dictionary like
    getClasses(2,2) = {0:(0,0),1:(0,1),2:(1,0),3:(2,0),4:(0,2),5:(1,1)}
  (is in no way sorted)
  '''

  class myKey:
    '''
    zur Berechnung bauen wir einen Baum an Zuständen auf. die Zustände sind
    gleichzeitig die Namen der Klassen. Z.B.:
      Name = (0,2,0,0)
    bedeutet, dass im zweiten v-Intervall 2/numberOfBricks Prozent der maximal mög-
    lichen Wellen vorzufinden sind, während in allen anderen Intevallen nichts
    los ist. Bei (0,0,2,0) wäre das ganze im dritten v-Intervall, also gleich
    viel Prozent an maximal möglichen Wellen, aber diesmal schneller.
    '''
    def __init__(self,father,ith_child,noB=0,secs=0):
      '''
      wenn erstes Objekt, dann secs, mS != 0
      wenn kind, dann == 0 und alles von Vater erben
      
      secs: number of sections
      noB:   number of bricks
      '''
      # default class variables (for tree start element if no father exists)
      self.name_book_ = {}
      self.name_ = [0] * secs
      self.name_ = tuple(self.name_)
      self.secs_ = secs
      self.noB_ = noB
      
      self.father_ = None
      self.children_ = None
            
      if secs==0: # child case:
        self.father_ = father
        self.name_book_ = self.father_.name_book_
        self.name_ = [el for el in self.father_.name_]
        self.name_[ith_child] += 1
        self.name_ = tuple(self.name_)
        self.secs_ = self.father_.secs_
        self.noB_ = self.father_.noB_
      
      if self.name_ not in self.name_book_.values(): # this state was not found before:
        self.name_book_[len(self.name_book_)] = self.name_
      if self.noB_ > np.sum(self.name_): # did not reach last level. continue with next level.
        self.children_ = [myKey(self,i) for i in range(self.secs_)]
        
    def getDict(self):
      print("Received {} classes.".format(len(self.name_book_)))
      return self.name_book_.copy()
  
  # Note: if a is the dictionary, you can get the backward map as b = {a[el]: el for el in a}
  return myKey(None,1,numberOfBricks,numberOfSections).getDict() # Teh first two arguments None and 1 could be anything.

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

def dictToArray(dicct,numberOfSections,behaviour='i'):
  '''
  Function converts the single amounts of waves for each velocity to an array with combined numbers per section.
  This is done for both, integer and probabilistic behaviour.
    * integer: f_i : \R^{N_v} --> \R^{N_s}
    * prob.:  f_p: DPD^{N_v} --> DPD^{N_s}, where DPD is some discrete prob. distr. (with at least one state).
  
  behaviour: integer behaviour is default.
  dicct: the amount of waves per contained velocity. Values are either \R^{N_v}
         or DPD^{N_v}, depending on use case (behaviour)
         Note: There are at max N_v velocities, not all of them are contained in every dicct. Also, N_v is unknown!
  numberOfSections: Here N_s.
    
  In this fuction, we have to decide, what happens if a velocity is exactly on a border. However,
  this is probably only a convention and does not effect the NNs qualitative behaviour.
  The implemented convention is:
    all intervals are half open:                '[)',
    except for the last one, which ist closed:  '[]'.
  This means, "border-velocities" usually get encounted for by the interval on its right side.
  Only on the last (upper) border, the border-velocity has to be counted to the left interval.
  For example, with (v_1,...v_6) = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), and N_s = 5,
  the situation looks like:
    '[v_1), [v_2), [v_3), [v_4), [v_5,v_6]'
    
  N_s (usually is, but) does not have to be smaller than N_v.
  
  ToDo: Implement probabilistic functinality.
  
  ToDo: get information of existing velocities! For now, we have to assume
  that 1.0 is the smallest, and 10.0 is the largest velocity. If this is not
  the case, ALL INTERVALS ARE PLACED WRONG!
  '''
  def getSectionForVelocity(v,borders):
    # determine interval number that v belongs to:
    return borders.searchsorted(v - (borders[1]-borders[0])) + int(v in borders) - int(v==borders[0] or v==borders[-1])
  
  if behaviour == 'p':
    raise Exception("ToDo: Implement probabilistic functinality!")
    
  retVal = [0]* numberOfSections
  borders=np.linspace(1.,10.,numberOfSections+1)
  for vel in dicct.keys():
    s_i = getSectionForVelocity(vel,borders)
    # add waves to section: (Note: for probabilistic behaviour this must be a DPD addition!)
    retVal[s_i] += dicct[vel]
  return retVal
  

def repairBricks(NNClass,NBricksOld,NBricksNew):
  '''
  return value may be of type np.array. Doesn't effect stuff outwards. Makes it here more convenient.
  '''
  retValue = [int(round(el*NBricksNew/NBricksOld)) for el in NNClass]
  if sum(retValue) > NBricksNew: # occurs seldom.
    print("NNClass = {},NBricksOld = {},NBricksNew = {}".format(NNClass,NBricksOld,NBricksNew))
    # Achtung: [2,8,0,2],12,4 -> [0.66,2.66,0,.66] -> [1,3,0,1] sum > 4!
    # können dort kürzen, wo der relative Fehler am kleinsten ist.
    print("Warning: Introduced errors due to insufficient NN-class conversion.")
    unrounded = np.array([el*NBricksNew/NBricksOld for el in NNClass])
    dismissedRetValue = np.array(retValue)
    indicesGT0 = np.array(np.where(dismissedRetValue>0)[0]) # this is guaranteed to be of len>0. looks a bit complicated, but does the job.
    relativeError = (dismissedRetValue[indicesGT0]-unrounded[indicesGT0]) / unrounded[indicesGT0] # may contain positive, as well as negative values. only positive are of interest. Neg. ones are related to a value that was rounded down, so further down-rounding would not make sense.
    relevantIndices = indicesGT0[np.where(relativeError>0)]
    errorList = sorted(np.unique(relativeError))
    currentErrorIndex = 0
    while sum(retValue) > NBricksNew:
      if errorList[currentErrorIndex] > 0:
        indicesToBeDeminished = np.array(relevantIndices[np.where(np.isclose(relativeError,errorList[currentErrorIndex]))[0]])[0]
        retValue[indicesToBeDeminished] -= 1
      currentErrorIndex += 1
  return retValue

def classify(list_of_dicts,training_opts):
  '''
  In this method, we sort all dictionaries in list_of_dicts to their respective class.
  There are different encodings, according to the users preferences:
    integer
      -> one-hot encoding
    probabilistic
      -> prob. distrib.
 
  
  For example: dict1 = {1.0: 3, 3.33: 1, 6.66: 0, 10.0: 0} -- relates - to - class -->  (4/mxmm,0) (with mxmm being the max number of waves)
  '''
  # get all options:
  trop_cl = training_opts['CL']
  trop_b = training_opts['B']
  trop_s = training_opts['S']    # the Number of Sections (the spectrum of speed is devided into)
  trop_enc = training_opts['GT'] # 'enc' stands for encoding
    
  # get the maximum number of waves that occurs in this list of dicts.
  maxW = 0
  length = len(list_of_dicts)
  count = np.ndarray([length],dtype=np.int32)
  for i in range(length):
    count[i] = 0
    for kel in list_of_dicts[i].keys():
      if (training_opts['GT']=='p'): # more complex data structure here:
        count[i] += int(max(list_of_dicts[i][kel].keys()))
      elif (training_opts['GT']=='i'):
        count[i] += int(list_of_dicts[i][kel])
    if count[i] > maxW:
      maxW = int(count[i])
      if type(maxW)!=type(2):
        print("typ von maxW Zeile 80 falsch")
  
  if trop_cl == 'd':
    # identify the case that yields direct behaviour: ==> numberOfBricks := maxWaves
    # overwrite the users (probably not explicitly specified key)
    trop_b = maxW
  
  # calculate how many Classes there exist
  NoC = getNumberOfClasses(trop_b,trop_s)
  classMapping = getClasses(trop_b,trop_s)
  backwardsMap = {classMapping[el]: el for el in classMapping}

  # hierin können wir die Verteilung der Input-Daten auf die existierenden Klassen schreiben.
  # aktuell nicht benötigt, bzw. keine Weiterverarbeitung.
  classDistribution = np.ndarray([NoC],dtype='int')*0
  #print(backwardsMap)

### Die metadaten enthalten beide versionen, p und i zeitlgleich als metaObject.pData_ und metaObject.iData_
  output = np.zeros([length,NoC]) # output[i] is in R^{NoC}, either a one hot encoding (all zeros, but one 1) or a probability distribution (adds to one).
  # WIR VERZICHTEN (aus Zeitgründen) vorerst auf die probabilistischen Daten (die nicht-one-hot-NNs)
  for i,el in enumerate(list_of_dicts):
    # convert dict to array, stateing the number of waves per wave speed section
    WavesPerSection = dictToArray(el,trop_s, trop_enc)
    # stretch/shrink if not 'direct' behaviour:
    if trop_b != maxW:
      WavesPerSection = repairBricks(WavesPerSection,maxW,trop_b)
    #print(WavesPerSection)
    className = tuple(WavesPerSection)
    classNumber = backwardsMap[className]
    classDistribution[classNumber] += 1
    output[i][classNumber] = 1
  
    #print(output)
  
  return output,maxW,NoC,classMapping

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
    grayscale_pictures,groundTruth = extractFrames(image_size,data_array[num],training_opts) # after this, groundTruth holds the raw dictionaries (states) that still need to be related to a class number / class name (done so in classify())
    end = filled_elements + grayscale_pictures.shape[0]
    x_temp[filled_elements:end] = grayscale_pictures
    filled_elements = end
    for el in groundTruth:
      groundtruths.append(el)
  
  #print(" ..classifying ..")
  # convert states to classes (one-hot encoding or prob.distr.)
  y_temp,maxWaves,number_of_classes,Map = classify(groundtruths,training_opts)
  
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
  ##  (Map,velIds_train,velIds_validation,groundtruths) makes the original waves still available for investigations
  return image_size,(x_train,y_train),(x_validation,y_validation),number_of_classes,(Map,velIds_train,velIds_validation,groundtruths,maxWaves)

  
def trainNeuralNetwork(pic_shape,training_data,valdtn_data,num_classes,training_opts):
  inputs = keras.Input(shape=(pic_shape[0],pic_shape[1],1))
  
### TODO: aufräumen. Struktur für Steuerung von Klassifizierung oder Regression ?!
###       1) Ordnung schaffen,  2) Regressionsmodelle entwerfen.
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
  elif training_opts['AR']==2:
    conv1 = layers.Conv2D(filters=32, kernel_size=(3, 3),padding='valid',activation=None)(x) # https://keras.io/examples/timeseries/timeseries_classification_from_scratch/ uses ReLU activation, but we already have values larger 0!
    
    conv2 = layers.Conv2D(filters=32, kernel_size=(1, 3),padding='valid',activation=None)(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    
    conv3 = layers.Conv2D(filters=32, kernel_size=(1, 3),padding='valid',activation=None)(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    
    gap = keras.layers.GlobalAveragePooling2D()(conv3)
    
    outputs = layers.Dense(num_classes, activation='softmax')(gap)
  #elif training_opts['AR']==3:
    
    
###
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.summary()
  
  ########## MÖGLICHE VARIATION: optimizers SGD, Adam, Adadelta, Adagrad, 
  ########## MÖGLICHE VARIATION: learning_rate
  #optim = keras.optimizers.RMSprop(learning_rate=1e-3)
  optim = keras.optimizers.Adadelta(learning_rate=1e-3)
  
  ########## MÖGLICHE VARIATION: keras.losses.CategoricalCrossentropy()
  lss = keras.losses.CategoricalCrossentropy() #keras.losses.MeanSquaredError()
  
  ########## MÖGLICHE VARIATION: metrics?
  model.compile(optimizer=optim,loss='mse', metrics = [keras.metrics.CategoricalAccuracy()])# loss = lss, metrics=['accuracy'])
  
  # compute NN and store the progress for eventual use, later.
  ########## MÖGLICHE VARIATION: batch_size
  history = model.fit(training_data[0],training_data[1],batch_size=1024,epochs=training_opts['EP'], validation_data = valdtn_data)
  
  return history, model

def autolabel(rects, ax):
  # Get y-axis height to calculate label position from.
  (y_bottom, y_top) = ax.get_ylim()
  y_height = y_top - y_bottom
  for rect in rects:
    height = rect.get_height()
    # Fraction of axis height taken up by this rectangle
    p_height = (height / y_height)
    # If we can fit the label above the column, do that;
    # otherwise, put it inside the column.
    if p_height > 0.95: # arbitrary; 95% looked good to me.
      label_position = height - (y_height * 0.05)
    else:
      label_position = height + (y_height * 0.01)
    ax.text(rect.get_x() + rect.get_width()/2., label_position,'%d' % int(height),ha='center', va='bottom')

def investigate(model,valData,step_size,noB,scene_mapping):
  #import scenario as sc
  x_validation,y_validation = valData
  classMap,velIds_train,velIds_validation,groundtruths,maxW = scene_mapping
  
  print(velIds_validation.shape)
  #list(map('{:.3f}%'.format,y_validation[num]))
  print("We will have a look on {} more or less random results.".format(len(np.arange(0,x_validation.shape[0],step_size))))
  for num in range(0,x_validation.shape[0],step_size):
    processed_data = model(x_validation[num:num+1])
    #print(processed_data)
    class_i = np.argmax(processed_data[0])
    reference_class = np.argmax(y_validation[num])
    #print("")
    print("NN Class predicted: {}, with {:.1f}% certainty.".format(class_i,100*np.max(processed_data[0])))
    P_ref = processed_data[0][reference_class]
    print("reference solution: {} (would have had {:.1f}% probability).".format(reference_class,P_ref*100))
    #print("control sum: {}".format(sum(processed_data[0])))
    #print(processed_data[0])
    #print("y_validation: \n {}".format(y_validation[num]))
    predictedClassName = classMap[class_i]
    TrueClassName = classMap[reference_class]
    originalDicct = groundtruths[velIds_validation[num][0]]
    wave_info = originalDicct,maxW
    if reference_class != 0 or class_i != 0:
      showClass(predictedClassName,noB,TrueClassName,wave_info)

def showClass(name,noOfBricks,truth=False,originalWaves=False):
  '''
    make a bar plot
    showing the specific class that is represented via its unique name
    The information noOfBricks is needed, since it is not contained in the class name.
    
    truth is optional, if used, it must be the true class name. If it is given,
    the produced plot compares the guess and the truth next to each other.
  '''
  
  granularity = 100. / noOfBricks
  sections = len(name)
  bar_heights = [el*granularity for el in name]
  labels = ["[{:.3}, {:.3})".format(i/sections*10. + (sections-i)/(sections),(i+1)/sections*10.+(sections-(i+1))/(sections)) for i in range(sections)]
  x = np.arange(len(labels))
  print("x is {}".format(x))
  width=0.99 # (default = 0.8)
  fig, ax = plt.subplots()
  ax.set_ylim(0.,100.)
  if truth:
    if type(truth) != type(name):
      raise Exception("truth must be a tuple of size {}".format(len(name)))
    elif len(truth) != len(name):
      raise Exception("truth must be a tuple of size {}".format(len(name)))
    width = width/2
    rects = ax.bar(x,bar_heights,1.9*width,label='guessed distribution',color='blue')#-width/2
    true_activity = [el*granularity for el in truth]
    true_rects = ax.bar(x,true_activity,2*width,label='ground thruth',color='green',alpha=0.4)# +width/2
    autolabel(rects,ax)
    autolabel(true_rects,ax)
  else:
    rects = ax.bar(x,bar_heights,width,label='fibre type distribution',color='blue')
    autolabel(rects,ax)
  if originalWaves:
    dicct,maxW = originalWaves
    velocities = np.array(sorted([key for key in dicct.keys()])) # every vel is unique and they are different in pairs
    numberV_i = [dicct[vel]/maxW*100 for vel in velocities]
    print("displayed velocities: {}".format(velocities))
    # map velocities from [1,10] to [-0.5, sections-0.5]
    mappedVelocities = ((velocities-1)/9*sections)-0.5
    peaks = ax.bar(mappedVelocities,numberV_i,width/10,label='original wave distribution',color='grey')
    
    
  ax.set_ylabel('muscle fibre activity [\%]')
  ax.set_xticks(x)
  ax.legend()
  #ax.bar_label(rects,padding=3)
  ax.set_xticklabels(labels)
  ax.set_xlabel('Fibre sEMG velocity [m/s]')
  fig.tight_layout()
  plt.show()
 
