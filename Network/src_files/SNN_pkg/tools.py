import sys
import random
import scipy
import torch
import time
import torch
from spikingjelly.activation_based import neuron, layer
from typing import *

def process_print(current_number, maximum):
    '''
    calling of this function should be start with 
    current_number=1, rather than 0
    '''
    cur_str=str(current_number)
    max_str=str(maximum)
    total_length=2*len(max_str)+1
    if current_number!=1:
        for i in range(total_length):
            print("\b", end='')
    space_num=len(max_str)-len(cur_str)
    for i in range(space_num):
        print(' ', end='')
    print(cur_str+'/'+max_str, end='')
    if current_number==maximum:
        print("")
    sys.stdout.flush()

class Logger():
    '''
    This is made for the record of STDP training process.
    Assistant class for STDPExe object defined in SNN_StdpModel.py
    '''
    def __init__(self, stdp_exe):
        self.exe=stdp_exe
        self.dir=stdp_exe.dir
    def write_log(self):
        '''
        Structure of the log file:
        timestamp, model structure, length of trainining dataset, time steps, Cl list
        '''
        with open(self.dir+'/STDP_log.txt', 'a') as file:
            timestamp=str(time.time())
            file.write(timestamp+'\n\n')
            modellog=str(self.exe.model)
            file.write(modellog+'\n\n')
            train_length=str(len(self.exe.train_data))
            file.write(train_length+'\n\n')
            file.write(str(self.exe.Cl_list))
            file.write('\n\n\n')
    def __getitem__(self, index)  -> List[str]:
        with open(self.dir+'/STDP_log', 'r') as file:
            content=file.read().split('\n\n\n')
            if index >=0:
                logline=content[-(index+2)]
            else:
                logline=content[-index-1]
            splitted=logline.split('\n\n')
        return splitted

def stochastic_for_stdp(amount :int, shape: tuple) -> List[Tuple]:
    '''
    Create schotastic data for STDP validation.
    '''
    data=torch.rand(amount,1,*shape)
    data[data>0.5]=1
    data[data<=0.5]=0
    ds=[(x, 1) for x in data]
    return ds

class SimpleDeap(torch.utils.data.Dataset):
    '''
    This is a class that packs preprocessed DEAP data and transfers them into spiking.
    '''
    def __init__(self, deap_dir: str):
        self.dir=deap_dir
        self.index=[] 
        for i in range(32):
            for j in range(40):
                self.index.append((i,j)) # The ith person's jth test
        random.shuffle(self.index)
    def __len__(self):
        return 32*40
    def __getitem__(self, i):
        person, test = self.index[i]
        file_index="0{}".format(person+1) if person<9 else str(person+1)
        mat=scipy.io.loadmat("{}/s{}.mat".format(self.dir, file_index))
        x=mat['data'][test]
        x=x[:14]
        y=mat['labels'][test][:2]
        x=torch.tensor(x, dtype=torch.float32)
        x[x>0]=1.
        x[x<=0]=0.
        y=torch.tensor(y, dtype=torch.float32)
        y[y<5]=0.
        y[y>=5]=1.
        return x,y
