# coding = utf-42

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import scenario_guts as ad
import meta_stuff as mt
import math

# TODO: Organizer umziehen nach data_creative_tools.
# in data_creator soll NICHT scenario eingebunden werden
class Organizer:
  def __init__(self):
    self.ids=0
  
  def get_id(self):
    self.ids += 1
    return self.ids - 1


# TODO init meta over scenario_guts. meta_stuff hier nicht einbinden!
class Scenario:
  
  def __init__(self,Elec,Len,DT,world):
    self.data_ = np.zeros(0)
    self.data_.shape = (0,Elec)
    self.meta_ = mt.Meta(0) # ad.initMeta(0)
    #self.initialized_ = False
    #self.finalized_ = False
    self.id_ = world.get_id()
    #print("Initializing Scenario {}".format(self.id_))
    #self.fin_ = np.array([],dtype=bool)
    self.number_of_electroids_ = Elec
    self.length_of_adhesive_array_ = Len
    self.time_between_sEMG_samples_ = DT
    self.IED_ = Len/(Elec-1) # inter electrode distance
    self.manager_ = world
   
          
  def __add__(self,other):
    # __add__ operator for two Scenario objects
    #
    # 1: only makes sense for measurements using the same measure-advise. Thus, otherwise ValueError.
    # 2: superposition with normalization --> f(x) \in [0,1]
    #
    # returns new Scenario object
    if (self.number_of_electroids_ != other.number_of_electroids_ or self.time_between_sEMG_samples_ != other.time_between_sEMG_samples_):
      raise ValueError("In _add_ scenarios ID{} and ID{}: Scenarios incompatible!".format(self.id_,other.id_))
    else:
      newObject = Scenario(self.number_of_electroids_,self.length_of_adhesive_array_,self.time_between_sEMG_samples_,self.manager_)
      if other.data_.shape[0] >= self.data_.shape[0]:
        newObject.data_ = other.data_.copy()
        newObject.data_[:self.data_.shape[0]] += self.data_.copy()
        newObject.meta_.set_size(other.data_.shape[0])
      else:
        newObject.data_ = self.data_.copy()
        newObject.data_[:other.data_.shape[0]] += other.data_.copy()
        newObject.meta_.set_size(self.data_.shape[0])
      # newObject.data_=np.minimum(newObject.data_,np.ones(newObject.data_.shape))
      mxmm = newObject.data_.max()
      #if mxmm = 0:
      #  nicht teilen
      if mxmm > 0 and mxmm != 1.0:
        # normieren:
        newObject.data_ /= mxmm
      #newObject.data_=
      newObject.meta_ = self.meta_ + other.meta_
      return newObject
    
  def set_size(self,T):
    mySize = self.data_.shape[0]
    if T < mySize:
      n_too_much = mySize - T
      print("Caution: cutting off the last {} time steps of Scenario ID{}.".format(n_too_much,self.id_))
      self.data_ = np.delete(self.data_,slice(T,mySize),axis=0)
      self.meta_.set_size(T)
      #self.fin_ = np.delete(self.meta_,slice(T,mySize),axis=0)
    elif T == mySize:
      print("Scenario ID{} has already size {}.".format(self.id_,mySize))
    else:
      #print("Appending zero-initialized time steps to Scenario ID{}.".format(self.id_))
      #print(self.data_.shape[1])
      self.data_ = np.vstack((self.data_,np.zeros((T-mySize,self.data_.shape[1]))))
      self.meta_.set_size(T)
          
  def add(self,smaller,start):
    '''
    fragmentale Methode - wird die noch irgendwo verwendet?
    '''
    # note, that here we only overlap at the end. we can not merge two small ones into a larger one as in Meta
    # but at least the 'smaller one' doesn't have to be actually smaller..
    mySize = self.data_.shape[0]
    smallerSize = smaller.data_.shape[0]
    if (self.data_.shape[1] != smaller.data_.shape[1]):
      print("In add() Scenarios ID{} and ID{}: Number of electroids don't match!".format(self.id_,smaller.id_))
    else:
      stop_small = start+smallerSize
      if stop_small > mySize:
        self.set_size(stop_small - mySize)
      self.data_[start:stop_small,:] += smaller.data_
      self.meta_.add(smaller.meta_,0,start)
  
  def __getitem__(self, key ) :
    '''
    with this method we can create new scenarios like
     >> new = s1[0:5] <<
    '''
    newObject = Scenario(self.number_of_electroids_,self.length_of_adhesive_array_,self.time_between_sEMG_samples_,self.manager_)
    #if isinstance( key, slice ) :
    newObject.data_ = self.data_[key,:]
    newObject.meta_ = self.meta_[key]
      #Get the start, stop, and step from the slice
    return newObject
  
  def accumulate(self,start,stop,n_waves,wave_speed,*args):
    # this function fills a specific part of a scenario, between start and stop.
    # N_time, thus must be == stop-start
    if (stop > self.data_.shape[0]):
      print("You must first enlarge the ndarrays of this Scenario via set_size()! (ID{})".format(self.id))
    else:
      data_plus,meta_plus = ad.toArray(n_waves,wave_speed,stop-start,self.number_of_electroids_,self.length_of_adhesive_array_,self.time_between_sEMG_samples_,*args)
      #self.add()
      #print("start = {}".format(start))
      #print("stop = {}".format(stop))
      #print("n_waves = {}".format(n_waves))
      #print("wave_speed = {}".format(wave_speed))
      self.data_[start:stop,:] += data_plus
      self.meta_.add(meta_plus,0,start)
        
  def compute_speed_variation(self,v,l,gap_vec,crit):
    #print("In compute_speed_variation: kind = {}".format(self.meta_.kind_))
    print("In compute_speed_variation: crit = {}".format(crit))
    # Wird auf ein Scenario angewendet.
    # Fügt mittels append ein Abfolge an Wellen an.
    # v: wave speed
    # l: wave length
    # gap_vec beinhaltet die Abstände zwischen den Wellen.
    #
    # berechne data und meta analytisch
    data_plus,meta_plus = ad.appendScenario(v,l,gap_vec,self.number_of_electroids_,self.length_of_adhesive_array_,self.time_between_sEMG_samples_,crit)
    mySize = self.data_.shape[0]
    #print("meta size: {}".format(self.meta_.size_))
    #print("data size: {}".format(mySize))
    #print("data_plus: \n {}".format(data_plus))
    self.set_size(mySize+data_plus.shape[0])
    self.data_[mySize:,:] += data_plus
    #print("size of meta data that ought to be __add__ed: {}".format(self.meta_.size_))
    # hängt meta_plus hinten an: self.meta_.add(meta_plus). Da aber mit self.set_size (zwei Zeilen oberhalb)
    # bereits das meta-Objekt vergrößert wurde, müssen wir die start-indizes beide gleich 0 setzen.
    self.meta_.add(meta_plus,0,0) 
    #self.meta_ = self.meta_ + meta_plus
    #print(type(self.meta_))
    #print("meta data size after '__add__': {}".format(self.meta_.size_))
    #self.fin_[mySize:] = True
  
  def maxWaves(self):
    max_waves = 0
    for ts in range(self.data_.shape[0]):
      ts_max = 0
      for key in self.meta_.pData_[ts].keys():
        #print(self.meta_.pData_[ts])
        #key is a specific speed. need to add all speed maxima
        ts_max += max(self.meta_.pData_[ts][key].keys())
      if ts_max > max_waves:
        max_waves = ts_max
    return max_waves

  def show(self,start=0,stop=0,colorbar=True):
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, ax = plt.subplots(1,1)
    if stop == 0:
      img = ax.imshow(self.data_[start:], cmap='gray', vmin=0, vmax=self.data_.max())
    else:
      img = ax.imshow(self.data_[start:stop], cmap='gray', vmin=0, vmax=self.data_.max())
    if colorbar:
      fig.colorbar(img)
    if start!=0:
      y_label_list = ['$t^{{now}}-{}dt$'.format(stop-start-i-1) for i in range(stop-start)]
      y_label_list[-1]='$t^{{now}}$'
      ax.set_yticks([i for i in range(stop-start)])
      ax.set_yticklabels(y_label_list,fontsize=16)
    plt.show()
    
  def compare(self,rows=0,cols=0,*args):
    number_of_scenarios = 1 + args.__len__()
    if number_of_scenarios==1:
      self.show()
    else:
      width_max = self.number_of_electroids_
      depth_max = self.data_.shape[0]
      for i,el in enumerate(args):
        d = el.data_.shape[0]
        w = el.number_of_electroids_
        if w > width_max:
          width_max = w
        if d > depth_max:
          depth_max = d
      
      if (rows*cols==0): # number of rows and cols needs to be determined.
        #approximate_ratio = int(depth_max / width_max)
        # plot configuration such that it looks nice...:
        #prettiest_form =  [depth_max,width_max]
        #ideal_ratio = 0
        surface_opt = 99999999999
        best_config = [0,0]
        for i in range(1,number_of_scenarios+1): # reihen
          must_have_cols = math.ceil(number_of_scenarios / i)
          j = must_have_cols
          if (2*j*width_max + 2*i*depth_max < surface_opt):
            surface_opt = 2*j*width_max +2*i*depth_max
            #print("(i,j)=({0},{1}) has {2}".format(i,j,surface_opt))
            best_config = [i,j]
            #ideal_ratio = i*depth_max / (j*width_max)
      
        # see, how we must set the figure size of the sub plots...:
        #prettiest_form = [self.data_.shape[0]/self.number_of_electroids_] # self at first
        #residuum = abs(prettiest_form[0]/prettiest_form[1]-ideal_ratio)
        #for i,el in enumerate(args):
        #  d = el.data_.shape[0]
        #  w = el.number_of_electroids_
        #  if abs(d/w-ideal_ratio) < residuum:
        #    residuum = abs(d/w-ideal_ratio)
        #    prettiest_form = [d,w]
        #    print(prettiest_form)
        (nrows,ncols) = best_config  # array of sub-plots
      else: # number of rows and cols is given by user
        (nrows,ncols) = (rows,cols)
        
      #print(ratio)
      #min(6/ncols,6/ratio/nrows)
      figsize = [min(width_max/min(width_max,depth_max),6),min(depth_max/min(width_max,depth_max),8)]     # figure size, inches. komischer weise in breite,höhe
      #print(figsize)
      # create figure (fig), and array of axes (ax)
      fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
      
      # plot simple raster image on each sub-plot
      for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        if i in range(number_of_scenarios):
          if i==0:
            img = self.data_
            ID = self.id_
          else:
            img = args[i-1].data_
            ID = args[i-1].id_
          #print("i={}".format(i))
          #print(img.max())
          axi.imshow(img, cmap='gray', vmin=0, vmax=img.max(),aspect='auto') # 1 was once 
          # get indices of row/column
          #rowid = i // ncols
          #colid = i % ncols
          # write row/col indices as axes' title for identification
          axi.set_title("ID "+str(ID))#Row "+str(rowid)+", Col "+str(colid)+": Scenario ID "+str(ID))

        # one can access the axes by ax[row_id][col_id] 
        # do additional plotting on ax[row_id][col_id] of your choice
        #ax[0][2].plot(xs, 3*ys, color='red', linewidth=3) 
        #ax[4][3].plot(ys**2, xs, color='green', linewidth=3)
      
      #plt.tight_layout(True)
      plt.show()
        
  def timeline(self,use_prob=False):
    N_time = self.data_.shape[0]
    time = np.linspace(0,N_time*self.time_between_sEMG_samples_,N_time)
    number_of_waves_per_speed_and_timei,number_of_waves_per_speed_and_timep, speed_list = self.meta_.get_vecs()
    n_waves=[]
    if use_prob:
      n_waves = number_of_waves_per_speed_and_timep
    else:
      n_waves = number_of_waves_per_speed_and_timei
    
    # create a cycler for distinct line styles of up to (8*4*5=160) different speeds:
    my_cycler = (cycler(color=['r', 'g', 'b', 'y','c','m','y','k']) * cycler(linewidth=[1,2,3,4]) * cycler(linestyle=['-', '--','-.', ':',(0,(3,5,1,5)),(0,(3,10,1,10,1,10))]))
    plt.rc('lines')
    plt.rc('axes', prop_cycle=my_cycler)
    plt.gca().invert_yaxis()
    
    for i in range(len(speed_list)):
      plt.plot(n_waves[i],time,label="v = "+str(speed_list[i]))
    
    plt.legend()
    plt.title("Scenario {}".format(self.id_))
    plt.ylabel("time [s]")
    plt.xlabel("number of waves")
    plt.show()
      
      
  def analyse(self):
    '''
    Visualisierung des Scenarios. es soll sichtbar sein, wann wieviele Wellen im System sind (einzelne keys und kumuliert)..
    rechts ist das sEMG Bild, links der Info-Graf.
    '''
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig, (ax2, ax1) = plt.subplots(nrows=1, ncols=2, sharey=False, gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle("Scenario {}".format(self.id_))
    
    # sEMG picture: 
    img = ax1.imshow(self.data_[:], cmap='gray', vmin=0, vmax=self.data_.max())
    ax1.set_ylabel("time step number")
    ax1.set_xlabel("electrode number")
    fig.colorbar(img,ax=ax1)
    
    # contained information:
    N_time = self.data_.shape[0]
    time = np.linspace(0,N_time*self.time_between_sEMG_samples_,N_time)
    number_of_waves_per_speed_and_timei,number_of_waves_per_speed_and_timep, speed_list = self.meta_.get_vecs()
    n_waves = number_of_waves_per_speed_and_timei
    cumulated_waves = []
    # all keys seperate
    my_cycler = (cycler(linestyle=['-', '--','-.', ':',(0,(3,5,1,5)),(0,(3,10,1,10,1,10))]) * cycler(linewidth=[1,2,3,4]) * cycler(color=['r', 'g', 'b', 'y','c','m','y','k']))
    plt.rc('lines')
    plt.rc('axes', prop_cycle=my_cycler)
    ax2.invert_yaxis()
    for i in range(len(speed_list)):
      ax2.plot(n_waves[i],time,label="v = "+str(speed_list[i]))
    
    # all together:
    for ts in range(len(n_waves[0])):
      cumulated_waves.append(sum(self.meta_.iData_[ts].values()))
    ax2.plot(cumulated_waves,time,label="cumulative")
    
    ax2.set_ylim([time[-1],time[0]])
    ax2.legend()
    ax2.set_ylabel("time [s]")
    ax2.set_xlabel("number of waves")
      
    
    plt.show()
      
      
      
      
      
      
      
      
      
      
      
      
      
      
      