'''
Zur besseren(?) Handhabung der main-Datei sind alle Commandline-Argumente
benannt und müssen nicht in einer spezifischen Reihenfolge stehen (aber immer
als key value - Paar direkt hintereinander). Sollen mehr Argumente eingepflegt
werden, muss das zum einen hier getan werden. Die Nutzung dieser neuen Argumente
muss entsprechend im späteren Programmablauf erfolgen.
'''


import sys

class InputPair:
  
  #sowas hilfreich: globalDict = {} ?
  
  def __init__(self,name,typo,description,validOptions=None):
    self.typo = typo
    if not isinstance(validOptions,type(None)):
      # only a finite set of options are allowed
      if len(validOptions) > 0:
        for option in validOptions:
          if type(option) != self.typo:
            try:
              option = self.typo(option)
            except:
              raise Exception("Types inconsistent. Could not convert to type {}.".format(typo))
        self.validOptions = tuple(validOptions)
      else:
        raise Exception("This usage doesn't make sense.")
    self.name = name
    self.xpln = description
    
  def getV(self):
    return self.value
    
  def setV(self,value):
    tmp = self.typo(value)
    try:
      if tmp in self.validOptions:
        self.value = tmp
      else:
        raise Exception("This value ({}) is not allowed for InputPair {}. Use one of {}.".format(value,self.name,self.validOptions))
    except AttributeError: # value may be anything
      self.value = tmp
      
  def getString(self):
    '''Depending on the options, that were used, a Network will be stored in a unique path.
       The day time will be added later as well.
       Here, THIS option adds its part to the path.
    '''
    retV = self.name + str(self.value)
    return retV
    
    
def getStandardArgumentList():
  Pair_1 = InputPair('AR',int, 'the neural network architecture',[0,1,2])
  Pair_2 = InputPair('GT',str, 'the kind of data interpretation. integer [or probabilistic -- not completely implemented]',['i','p'])
  Pair_3 = InputPair('CL',str, 'the way classes are defined. manual/granular mode or auto/direct mode OR not at all/regression task',['m','g','a','d','n','r']) # wenn das hier n oder r ist, dann braucht man gewisse andere nicht
  Pair_4 = InputPair('S', int, 'the number of sections to be used (exists for classification and regression)')
  Pair_5 = InputPair('B', int,'the number of bricks to be used if CL=g or m. Otherwise autodetermined, however, please mention something')
  Pair_6 = InputPair("EP",int, 'the number of epochs to be used for the training')
  Pair_7 = InputPair("TS",int, 'the number of time steps that define a picture')
  Pair_8 = InputPair("DT",str, 'the filename of the training data')
  return {Pair_1.name: Pair_1, Pair_2.name: Pair_2, Pair_3.name: Pair_3, Pair_4.name: Pair_4, Pair_5.name: Pair_5, Pair_6.name: Pair_6, Pair_7.name: Pair_7, Pair_8.name: Pair_8}
 
def getRegressionArgumentList():
  Pair_1 = InputPair('AR',int, 'the neural network architecture',[3,4])
  Pair_2 = InputPair('GT',str, 'the kind of data interpretation. integer [or probabilistic -- not completely implemented]',['i','p'])
  Pair_3 = InputPair('CL',str, 'the way classes are defined. manual/granular mode or auto/direct mode OR not at all/regression task',['m','g','a','d','n','r']) # wenn das hier n oder r ist, dann braucht man gewisse andere nicht
  Pair_4 = InputPair('S', int, 'the number of sections to be used (exists for classification and regression)')
  #Pair_5 = InputPair('B', int,'the number of bricks to be used if CL=g or m. Otherwise autodetermined, however, please mention something')
  Pair_6 = InputPair("EP",int, 'the number of epochs to be used for the training')
  Pair_7 = InputPair("TS",int, 'the number of time steps that define a picture')
  Pair_8 = InputPair("DT",str, 'the filename of the training data')
  return {Pair_1.name: Pair_1, Pair_2.name: Pair_2, Pair_3.name: Pair_3, Pair_4.name: Pair_4, Pair_6.name: Pair_6, Pair_7.name: Pair_7, Pair_8.name: Pair_8}  

def errorAndExit(lst=[]):
  '''
  Wenn das main Programm nicht ordnungsgemäß aufgerufen wird, informieren wir
  den Nutzer, was von ihm erwartet wird.
  '''
  print("Please specify arguments as follows: 'python3 main.py <key_1 value_1> ... <key_7 value_7>'. For example:")
  print(" $ python3 main.py AR 0 GT i CL a S 1 B 1 DT DATA/creator_001/file.pickle EP 100 TS 5\n")
  if len(lst) > 0:
    print("Valid keys for this example are:")
    for el in lst:
      print("'{}' ({})".format(el.name,el.type, el.xpln))
      if el.validOptions:
        print("  - valid options are {}".format(el.validOptions))
  print()
  print("Program terminating.")
  sys.exit(1)
  
def  printArgs(args):
  print('=================================================================================')
  for el in args:
    print("  {},  {} = {}".format(el.xpln, el.name, el.getV()))
  print('=================================================================================')
  
def process(arguments):
  N = len(arguments)  
  
  print(arguments)
  if (len(sys.argv) != 2*N + 1):
    ii=0
    for el in sys.argv:
      print("Argument {}: {}".format(ii,el))
      ii+=1
    errorAndExit()
  temp = sys.argv[1::2]
  # pair the arguments:
  arg_map = dict((el,0) for el in temp)
  for i,el in enumerate(temp):
    key=str(el)
    arg_map[key] = sys.argv[2+2*i]
  
  for el in arguments.keys():
    if el in arg_map.keys():
      print(el)
      print(arg_map[el])
      arguments[el].setV(arg_map[el])
  
  # control if all mandatory arguments are given
  ###TODO: 
  
#def processArguments():
#  Anzahl_Parameter = 8
#  if (len(sys.argv) != Anzahl_Parameter*2+1):
#    ii=0
#    for el in sys.argv:
#      print("Argument {}: {}".format(ii,el))
#      ii+=1
#    errorAndExit()
#  temp = sys.argv[1::2]
#  # random order:
#  arguments=dict((el,0) for el in temp)
#  for i,el in enumerate(temp):
#    key=str(el)
#    arguments[key] = sys.argv[2+2*i]
#  ## integer values:
#  if ('AR' in arguments):
#      arguments['AR'] = int(arguments['AR'])
#  if ('S' in arguments):
#  # S=0: NN unterscheidet klassen nach anzahl an wellen
#  # S>0: NN unterscheidet klassen nach geschwindigkeiten und bemisst diese anteilig
#      #if (not int(arguments['S'])):
#      #    print()
#      #    print("Number of Sections can't be zero. It must at least be 1.")
#      #    print("Program terminating.")
#      #    sys.exit(1)
#      arguments['S'] = int(arguments['S'])
#  if ('EP' in arguments):
#      arguments['EP'] = int(arguments['EP'])
#  if ('TS' in arguments):
#      arguments['TS'] = int(arguments['TS'])
#  ## char values:
#  if ('GT' in arguments):
#    if arguments['GT'] not in ['i','p']:
#      print("{} is not a valid value for 'GT', the kind of data interpretation. Must be 'i' (integer) or 'p' (probabilistic).".format(arguments['GT']))
#      print("Program terminating.")
#      sys.exit(1)
#  if ('CL' in arguments):
#    if arguments['CL'] not in ['g','d','a','m','n','r']:
#      print("{} is not a valid value for 'CL', the way classes are defined. Must be 'g' or 'm' (granular) or 'd' or 'a' (direct) or 'n' or 'r' (no classes = regression).".format(arguments['CL']))
#      print("Program terminating.")
#      sys.exit(1)
#    elif (arguments['S']==0 and (arguments['CL'] in ['g','m'])):
#      print("With granular classes (CL='g' or 'm'), the number of sections, S, must be > 0.".format(arguments['CL']))
#      print("Program terminating.")
#      sys.exit(1)
#    if arguments['CL']=='m': # use m (manual) only for user context. is g inside
#      arguments['CL']='g'
#    if arguments['CL']=='a': # use a (auto) only for user context. is d inside
#      arguments['CL']='d'
#  #if ('DT' in arguments):
#  #  if file does not exist
#  ## float values:
#  if ('B' in arguments): # this argument could have a different meaning when using regression..
#    if arguments['B']==0:
#      if not (arguments['CL']=='n' or arguments['CL']=='r'):
#        print("With granular classes (CL='g' or 'm'), the number of bricks, B, must be > 0.".format(arguments['CL']))
#        print("Program terminating.")
#        sys.exit(1)
#      else:
#        print("Command line option key B given although you chose regression mode. No meaning associated to this option, yet.".format(arguments['CL']))
#    else:
#      arguments['B'] = int(arguments['B'])
#  printArgs(arguments)
#  return arguments