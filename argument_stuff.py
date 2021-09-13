 
import sys

def errorAndExit():
  print("Please specify the following arguments as: python3 main.py <key_1 value_1> ... <key_7 value_7>")
  print("The keys are:")
  print('  "AR"  (int): the neural network architecture')
  print('  "GT" (char): the kind of data interpretation. integer (i) or probabilistic (p)')
  print('  "CL" (char): the way classes are defined. granular (g) or direct (d)') 
  print('  "S"   (int): the number of sections to be used')
  print('  "G" (float): the granularity if CL=g, otherwise meaningless(?!)')
  print('  "EP"  (int): the number of epochs to be used for the training')
  print('  "TS"  (int): the number of time steps that define a picture')
  print('  "DT" (char): the filename of the training data')
  print()
  print("Program terminating.")
  sys.exit(1)
  
def  printArgs(args):
  print('=================================================================================')
  print("  the neural network architecture,   AR = {}".format(args['AR']))
  print("  the kind of data interpretation,   GT = '{}'".format(args['GT']))
  print("  the way classes are defined,       CL = '{}'".format(args['CL'])) 
  print("  the number of sections to be used,  S = {}".format(args['S']))
  print("  the granularity,                    G = {}".format(args['G']))
  print("  the number of epochs to be used,   EP = {}".format(args['EP']))
  print("  the number of time steps per pic., TS = {}".format(args['TS']))
  print("  the filename of the training data, DT = '{}'".format(args['DT']))
  print('=================================================================================')
  
def processArguments():
  Anzahl_Parameter = 8
  if (len(sys.argv) != Anzahl_Parameter*2+1):
    ii=0
    for el in sys.argv:
      print("Argument {}: {}".format(ii,el))
      ii+=1
    errorAndExit()
  temp = sys.argv[1::2]
  # random order:
  arguments=dict((el,0) for el in temp)
  for i,el in enumerate(temp):
    key=str(el)
    arguments[key] = sys.argv[2+2*i]
  ## integer values:
  if ('AR' in arguments):
      arguments['AR'] = int(arguments['AR'])
  if ('S' in arguments):
  # S=0: NN unterscheidet klassen nach anzahl an wellen
  # S>0: NN unterscheidet klassen nach geschwindigkeiten und bemisst diese anteilig
      #if (not int(arguments['S'])):
      #    print()
      #    print("Number of Sections can't be zero. It must at least be 1.")
      #    print("Program terminating.")
      #    sys.exit(1)
      arguments['S'] = int(arguments['S'])
  if ('EP' in arguments):
      arguments['EP'] = int(arguments['EP'])
  if ('TS' in arguments):
      arguments['TS'] = int(arguments['TS'])
  ## char values:
  if ('GT' in arguments):
    if arguments['GT'] not in ['i','p']:
      print("{} is not a valid value for 'GT', the kind of data interpretation. Must be 'i' (integer) or 'p' (probabilistic).".format(arguments['GT']))
      print("Program terminating.")
      sys.exit(1)
  if ('CL' in arguments):
    if arguments['CL'] not in ['g','d']:
      print("{} is not a valid value for 'CL', the way classes are defined. Must be 'g' (granular) or 'd' (direct).".format(arguments['CL']))
      print("Program terminating.")
      sys.exit(1)
    elif (arguments['S']==0 and arguments['CL']=='g'):
      print("With granular classes (CL='g'), the number of sections, S, must be > 0.".format(arguments['CL']))
      print("Program terminating.")
      sys.exit(1)
  #if ('DT' in arguments):
  #  if file does not exist
  ## float values:
  if ('G' in arguments):
    if arguments['G']==0:
      print("With granular classes (CL='g'), the granularity, G, must be > 0.".format(arguments['CL']))
      print("Program terminating.")
      sys.exit(1)
    else:
      arguments['G'] = float(arguments['G'])
  printArgs(arguments)
  return arguments