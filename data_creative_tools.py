
import scenario as sc
import math
import numpy as np

def get_gaps(N,s,l):
  if (N%2!=1 or N < 1):
    return False
  #
  # returns a numpy array with gaps of sizes in [s,l]
  # the outer elements are larger than the inner ones, decreasing until the middle one is the smallest.
  #
  # for N = 7 it shall return np.array[l,100*s,10*s,s,10*s,100*s,l]
  # 
  else:
    factor = pow(l/s,2/(N-1))
    gap_lengths = np.ndarray(N)
    gap_lengths[int((N+1)/2):] = np.array([s*pow(factor,i) for i in range(1,int((N+1)/2))])
    gap_lengths[int((N-1)/2)]  = s
    gap_lengths[:int((N-1)/2)] = np.array([s*pow(factor,i) for i in reversed(range(1,int((N+1)/2)))])
    #print(gap_lengths)
    return gap_lengths

def get_scenarios_with_fixed_wavelength(N_speeds,lambda_,world,dt=1/2048,N_electroids=16,len_=0.075,crit=0.0):
  scenario = []
  id = 0
  max_size = 0
  gap_vec = np.arange(0)
  for speed in np.linspace(1.0,10,N_speeds):
    if id==0:
      #AllOthersShouldBeLessThanThisIfPossible = math.ceil(2*(len_+lambda_)/speed/dt)+math.ceil((len_+2*lambda_+len_+lambda_)/speed/dt)
      gap_vec = get_gaps(7,lambda_/250,len_)
      max_size = math.ceil(((len(gap_vec)+1)*lambda_+gap_vec.sum()+len_)/speed/dt)+1
      #print("{} waves".format(len(gap_vec)+1))
      #print("make a total of {} time steps".format(math.ceil((len(gap_vec)+1)*lambda_/speed/dt)))
      #print("and then {} more for the gaps, so that's why ..".format(math.ceil(gap_vec.sum()/speed/dt))) # math.ceil(gap_vec.sum()/speed/dt)
      #print("I thought that the size for the coarsest scenario was {}.".format(max_size))
      #print(math.ceil(len_/speed/dt))
      #print(math.ceil(lambda_/speed/dt))
      #print(math.ceil(lambda_/250/speed/dt))
      scenario.append(sc.Scenario(N_electroids,len_,dt,world))
      scenario[id].compute_speed_variation(speed,lambda_,gap_vec,crit)
      print("computing with speed {} and wavelength {}.".format(speed,lambda_))
      max_size = scenario[id].meta_.size_
      #print("But actually the size for the coarsest scenario is {}.".format(max_size))
    else:
      number_of_gaps_to_use = 7
      temp_vec = get_gaps(number_of_gaps_to_use,lambda_/250,len_)
      temp_size = math.ceil(((len(gap_vec)+1)*lambda_+gap_vec.sum()+len_)/speed/dt)+1
      while temp_size <= max_size:
        # this is a proven configuration:
        gap_vec = temp_vec
        #actual_size = temp_size
        #prepare to test the next larger one:
        number_of_gaps_to_use += 2
        temp_vec = get_gaps(number_of_gaps_to_use,lambda_/250,len_)
        temp_size = math.ceil(((len(temp_vec)+1)*lambda_+temp_vec.sum()+len_)/speed/dt)+1
      scenario.append(sc.Scenario(N_electroids,len_,dt,world))
      scenario[id].compute_speed_variation(speed,lambda_,gap_vec,crit)
      print("computing with speed {} and wavelength {}.".format(speed,lambda_))
    id += 1
  return scenario
  