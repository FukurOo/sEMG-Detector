import numpy as np
import misc

class Meta:

  def __init__(self,size):
    self.iData_ = [{} for k in range(size)] # interpreted (integer) numbers of waves.
    self.pData_ = [{} for k in range(size)] # probabilistic point of view regarding the wave numbers
    self.size_ = size
    # kind = 0 means i
    # kind = 1 means p 

  def __add__(self,other):
    newObject=Meta(max(self.size_,other.size_))
    #print(type(self))
    #print(type(other))
    for i in range(newObject.size_):
      # it is not given that self.xData_[i] exists
      # it is not given that other.cData_[i] exists
      if self.size_ > i and other.size_ > i:
        for speed in other.iData_[i].keys(): # die keys in iData_ und pData_ sind die selben!
          if speed in self.iData_[i].keys():
           # integer: 
            newObject.iData_[i][speed] = self.iData_[i][speed] + other.iData_[i][speed]
           # probabilistic:
            # the fusion of the two probabilistic dicts is non trivial.
            # I should receive a larger dict, just like:
            # {4: 0.7, 5:0.3, 6:0.0} + {0:0.1, 1: 0.4, 2:0.5} = {4:P4, 5:P5, 6:P6, 7:P7, 8:P8,}
            # 
            # P4 = 0.7*0.1                      = 0.07
            # P5 = 0.7*0.4 + 0.3*0.1            = 0.31
            # P6 = 0.7*0.5 + 0.3*0.4 + 0.0*0.1  = 0.47
            # P7 =           0.3*0.5 + 0.0*0.4  = 0.15
            # P8 =                     0.0*0.5  = 0.00
            # SUM(Pi) =                           1.00
            #
            # um Platz und Rechenoperationen zu sparen, könnte man die trailing zeros fallen lassen.
            # wenn man innere zeros fallen lässt, dann bekommt man probleme. Außerdem ergibt das von der Wahrscheinlichkeits-
            # Verteilung keinen Sinn, weil minima nur außen anliegen dürfen. Zu Beginn werden zeros ohnehin nicht produziert.
            new_dict = {}
            # inefficient cycling through the cases.. looping N² times in stead of N times. however, only low N expected ...:
            for wave_number1 in self.pData_[i][speed].keys():
              for wave_number2 in other.pData_[i][speed].keys():
                N_new = wave_number1 + wave_number2
                if N_new in new_dict.keys():
                  new_dict[N_new] += self.pData_[i][speed][wave_number1]*other.pData_[i][speed][wave_number2]
                else: # automated conditional creation of new  key. Only works with '=', not with '+='.
                  P_N = self.pData_[i][speed][wave_number1]*other.pData_[i][speed][wave_number2]
                  if P_N: # wenn nicht 0
                    new_dict[N_new] = P_N
            newObject.pData_[i][speed] = new_dict
          else: # automated creation of new  key. Only works with '=', not with '+='.
            newObject.iData_[i][speed] = other.iData_[i][speed]
            newObject.pData_[i][speed] = other.pData_[i][speed]
        for speed in self.iData_[i].keys():
          # make sure we did not already consider them:
            if speed not in other.iData_[i].keys():
              newObject.iData_[i][speed] = self.iData_[i][speed]
              newObject.pData_[i][speed] = self.pData_[i][speed]
      elif self.size_ > i:
        newObject.iData_[i] = self.iData_[i]
        newObject.pData_[i] = self.pData_[i]
      elif other.size_ > i:
        newObject.iData_[i] = other.iData_[i]
        newObject.pData_[i] = other.pData_[i]
      else:
        raise Exception("This case should not occurr!")
    return newObject
      
  def __getitem__(self, key ) :
    '''
    with this method we can create new meta objects like
     >> new = m1[3:5] <<
    '''
    newObject=[]
    if isinstance(key,slice):
      newObject = Meta(key.stop-key.start)
    elif isinstance(key,int):
      newObject = Meta(1)
    newObject.iData_ = self.iData_[key]
    newObject.pData_ = self.pData_[key]
    #if isinstance( key, slice ) :
    return newObject  
  
  def add(self,other, start1=0, start2=-1):
    # angepasst für fusionierte probabilistic und integer meta data.
    # here we have two objects that we merge into a new one. The new then replaces self.
    #           other:
    #    |      (x,y,z)
    #                      +        self:
    #    |                          (a,b,c)
    #            ^                   ^
    #           s2    ^             s1     ^
    #               size2                size1=old size (="3")
    #     new 0           =             new size (="16")
    #      Y                               Y
    #    |(*,*,*,x,y,z,*,*,*,*,*,*,*,a,b,c)
    #     new self
    # 
    # Note that self and other can overlap!
    #
    #print("self size {}".format(self.size_))
    #print("other size {}".format(other.size_))
        
    if (start2==-1):
      start2=self.size_
    
    new_size=max(start1+self.size_,start2+other.size_)
    #print("new meta size: {}".format(new_size))
    iTemp =[{} for k in range(new_size)]
    pTemp =[{} for k in range(new_size)]
    
    for i in range(self.size_):  # übertrage alt (self)  in neu
      for speed in self.iData_[i].keys():
        iTemp[start1+i][speed] = self.iData_[i][speed]
        pTemp[start1+i][speed] = self.pData_[i][speed]
    for i in range(other.size_): # übertrage other in neu
      for speed in other.iData_[i].keys():
        if speed in iTemp[start2+i].keys():    # wenn schon drin, addiere
         # integer:
          iTemp[start2+i][speed] += other.iData_[i][speed]             # funktioniert NICHT für probabilistic 
         # prob:                                                       # deshalb anders (wie auch in __add__())
          new_dict = {}
          # inefficient cycling through the cases.. looping N² times in stead of N times. however, only low N expected ...:
          for wave_number1 in self.pData_[i][speed].keys():
            for wave_number2 in other.pData_[i][speed].keys():
              N_new = wave_number1 + wave_number2
              if N_new in new_dict.keys():
                new_dict[N_new] += self.pData_[i][speed][wave_number1]*other.pData_[i][speed][wave_number2]
              else: # automated conditional creation of new  key. Only works with '=', not with '+='.
                P_N = self.pData_[i][speed][wave_number1]*other.pData_[i][speed][wave_number2]
                if P_N: # wenn nicht 0
                  new_dict[N_new] = P_N
          pTemp[start2+i][speed] = new_dict
        else:                                 # wenn nicht, erzeuge
          iTemp[start2+i][speed] = other.iData_[i][speed]
          pTemp[start2+i][speed] = other.pData_[i][speed]    
    self.iData_ = iTemp    
    self.pData_ = pTemp
    self.size_ = new_size
    
  def set_size(self,new_size):
    if new_size < self.size_:
      del self.iData_[new_size:]
      del self.pData_[new_size:]
      self.size_= new_size
    elif new_size > self.size_:
      self.iData_ = self.iData_ + [{} for k in range(new_size-self.size_)]
      self.pData_ = self.pData_ + [{} for k in range(new_size-self.size_)]
      self.size_ = new_size
      
  def get_vecs(self):
    ###
    # :: Returns a matrix containing the [number of occurrances per time step = row] of [all speeds that exist in this scenario = column]
    ###
    # how does the integer perspective differ from the probabilistic point of view?
    # integer can be displayed directly. probabilistic has to be interpreted before, or has to be displayed more complex.
    #
    ###
    #n_keys = 0
    accumulated_speeds = [] # lists all speeds that exist in this scenario 
    # # # right now, they are not sorted !!
    for i in range(self.size_):
      for speed in self.iData_[i].keys():
        if speed not in accumulated_speeds:
          accumulated_speeds.append(speed)
    return_iVector = np.zeros((len(accumulated_speeds),self.size_))
    return_pVector = return_iVector.copy()
    accumulated_speeds.sort()
    for i in range(self.size_):
      for speed in self.iData_[i].keys():
        return_iVector[accumulated_speeds.index(speed)][i] = self.iData_[i][speed]
        return_pVector[accumulated_speeds.index(speed)][i] = misc.interpret(self.pData_[i],speed)
    return return_iVector,return_pVector,accumulated_speeds