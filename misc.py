import matplotlib.pyplot as plt
import numpy as np

def show(data):
  reshaped = np.ndarray([data.shape[0],data.shape[1]])
  reshaped[:,:] = data[:,:,0] 
  plt.imshow(reshaped, cmap='gray', vmin=0, vmax=reshaped.max())
  plt.show()

def interpret(dicct,key):
  # probabilistic dicts contain prob distributions.
  # for a certain speed, key, we want a interpreted version of how many waves are there. this is float, not in. so it is a guess:
  #   dicct =  {4.6: {7: 0.3, 8: 0.6, 9: 0.1}, 6.2: {4: 0.08, 5: 0.12, 6: 0.8}}
  #   key   =   4.6
  #   returns 7.8 = 2.1 + 4.8 + 0.9 = 7*0.3 + 8*0.6 + 9*0.1
  if key not in dicct.keys():
    print("error in interpret. Key not found")
    return -1
  sum_ = 0
  for sub_key in dicct[key].keys():
    sum_ += sub_key * dicct[key][sub_key]
  return sum_