import sys
import random
import scipy
import torch
import time
import torch
import numpy as np
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
    This is a class that packs preprocessed DEAP data.
    Preferred store format for each kind of data:
    1. None-spiking preprocessed deap:
        DEAP
        —— .mat (including keys "data", "labels")
    2. BSA:
        DEAP
        —— signals
            —— .npy
        —— labels
            —— .npy
    '''
    def __init__(self, deap_dir: str, **argv):
        '''
        in_memory: the amount to stored into the memory.
        deap_dir: the directory of the folder that obeys preferred store format
        Parameters in argv:
        1. index
        2. memory_num = 1
        3. channel_amount = 32
        4. mode = 'spiking' (or prep, origin, BSA)
        5. d_label = "basic"
        '''
        self.argv=argv
        if "channel_amount" in argv:
            self.channel_amount=argv["channel_amount"]
        else:
            self.channel_amount=32
        self.dir=deap_dir
        if "index" in argv:
            self.index=argv["index"]
        else:
            self.index=[] 
            for i in range(32):
                for j in range(40):
                    self.index.append((i,j)) # The ith person's jth test
            random.shuffle(self.index)
        if "d_label" in argv:
            self.d_label=argv["d_label"]
        else:
            self.d_label="basic"
        with open("index_{}.txt".format(self.d_label), 'w') as file:
            file.write(str(self.index))
        if "memory_num" in argv:
            self.memory_num=argv["memory_num"]
        else:
            self.memory_num=1
        if "mode" in argv:
            self.mode=argv["mode"]
        else:
            self.mode="spiking"
        self.memory=dict()
    def split(self, test_ratio=0.1):
        test_len=int(len(self)*test_ratio)
        train_len=len(self)-test_len
        train_data=SimpleDeap(self.dir, **self.argv, index=self.index[:train_len], d_label="train")
        test_data=SimpleDeap(self.dir, **self.argv, index=self.index[train_len:], d_label="test")
        return train_data, test_data
    def __len__(self):
        return len(self.index)
    def __getitem__(self, i):
        person, test = self.index[i]
        if person in self.memory:
            mat=self.memory[person]
        else:
            if len(self.memory.keys()) >= self.memory_num:
                first_key=list(self.memory.keys())[0]
                del self.memory[first_key]
            if self.mode!="BSA":
                file_index="0{}".format(person+1) if person<9 else str(person+1)
                mat=scipy.io.loadmat("{}/s{}.mat".format(self.dir, file_index))
            else:
                mat=np.load("{}/signals/BSA_signal_{}.npy".format(self.dir, person))
            self.memory[person]=mat
        if self.mode=="BSA":
            x=mat[test]
        else:
            x=mat['data'][test]
        x=x[:self.channel_amount]
        x=torch.tensor(x, dtype=torch.float32)
        if self.mode=="BSA":
            mat=np.load("{}/labels/BSA_labels_{}.npy".format(self.dir, person))
            y=mat[test]
        else:
            y=mat['labels'][test][:2]
        y=torch.tensor(y, dtype=torch.float32)
        if self.mode=="spiking":
            x[x>0]=1.
            x[x<=0]=0.
        elif self.mode=="prep":
            x=x.abs()
            x=x/x.max()
        else:
            x=x
        if self.mode!="BSA":
            y[y<5]=0.
            y[y>=5]=1.
        x=x.to(dtype=torch.float32)
        y=y.to(dtype=torch.float32)
        return x,y

