import sys
import pickle
import scenario
import basics


# Convert data from the old format to the new one with a dictionary
def convert_data_old(data):
    """
    data should be a list of scenarios
    """
    
    velocities = set()
    # case : data_creator_002
    if isinstance(data[0], scenario.Scenario):
        for sc in data:
            for v_array in sc.meta_.get_vecs()[1]:
                for v in v_array:
                        if v > 0:
                            velocities.add(v)
    # case : data_creator_001
    else:       
        for inner_lists in data :
            for sc in inner_lists:
                for v_array in sc.meta_.get_vecs()[1]:
                     for v in v_array:
                        if v > 0:
                            velocities.add(v)
                            
    velocities = list(velocities)
    velocities.sort()
    max_waves = 0
    waveLengths = 0
    new_data_format = {"content":data, "maxW":max_waves,"velocities":velocities, "wLengths": waveLengths}
    return new_data_format
    #except:
    #    print("Error, could not convert data.")
    
def convert_data(data):
  """
  data should be a list of scenarios
  
  this function extracts some valuable information out of a bunch of scenarios.
  """
  # to get all velocites that occur:
  velocities = []
  # to get the maximum amount of concurrently ocurring waves, regardless of their velocity:
  i_dicts = []
  p_dicts = []
  if isinstance(data[0], scenario.Scenario):
    for scene in data:
      #maxW = max(scene.maxWaves(),maxW)
      print("\n Fall 1 ")
      print(scene.meta_.iData_)
      print()
      for i,dct in enumerate(scene.meta_.iData_):
        i_dicts.append(dct)
        p_dicts.append(scene.meta_.pData_[i])
        for speed in dct.keys():
          if speed not in velocities:
            velocities.append(speed)
  else:
    for inner_lists in data:
      for scene in inner_lists:
        #maxW = max(scene.maxWaves(),maxW)
        for i,dct in enumerate(scene.meta_.iData_):
          i_dicts.append(dct)
          p_dicts.append(scene.meta_.pData_[i])
          for speed in dct.keys():
            if speed not in velocities:
              velocities.append(speed)
            
  velocities.sort()
  
  #intermezzo to get max waves:
  global_maxi = basics.maxwaves(i_dicts,waves_counted_probabilistic=False)
  del i_dicts
  global_maxp = basics.maxwaves(p_dicts,waves_counted_probabilistic=True)
  del p_dicts
  if global_maxi != global_maxp:
    print("\nINFORMATION: global_max of wave count is different for\n upper bound ({}) - and hard criterium ({}) way of counting.".format(global_maxi,global_maxp))
  maxW = (global_maxi,global_maxp) # war max(,)
    
  new_data_format = {'content': data.copy(), 'maxW': maxW,'velocities': velocities, "wLengths": None}
  return new_data_format

# Converts pickle-file to the new format if format is old and saves the new version
def convert_file(file):
    with open(file, "rb") as f:
        old_data = pickle.load(f)
        try: 
            if(type(old_data) is dict):
                print("File has already new format. (expected behaviour)")
            else:
                new_data_format = convert_data(old_data)  
                file_name = f.name
                pickle.dump(new_data_format, open(file_name, "wb"))
                pickle.dump(old_data,open(file_name+'old',"wb"))
                print("File was successfully changed. However, 'wLengths' could not be recovered, and thus, could not be documented.")
        except:
            print("Unknown format or path.")
    f.close()
    
    
# only run, if module is explicitly called 
if __name__ == '__main__':
    path = str(sys.argv[1])
    file = open(path, 'rb')
    convert_file(file)