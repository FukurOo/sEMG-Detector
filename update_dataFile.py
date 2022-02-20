import sys
import pickle
import scenario


# Convert data from the old format to the new one with a dictionary
def convert_data(data):
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
    



# Converts pickle-file to the new format if format is old and saves the new version
def convert_file(file):
    with open(file, "rb") as f:
        old_data = pickle.load(f)
        try: 
            if(type(old_data) is dict):
                print("File has already new format.")
            else:
                new_data_format = convert_data(old_data)    
                file_name = f.name
                pickle.dump(new_data_format, open(file_name, "wb"))
                print("File was successfully changed.")
        except:
            print("Unknown format or path.")
    f.close()
    
    
# only run, if module is explicitly called 
if __name__ == '__main__':
    path = str(sys.argv[1])
    file = open(path, 'rb')
    convert_file(file)