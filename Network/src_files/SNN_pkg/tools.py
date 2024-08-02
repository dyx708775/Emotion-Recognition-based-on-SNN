# Xiangnan Zhang 2024, School of Future Technologies, Beijing Institute of Technology
# modified: 2024.7.18
# Dependencies: PyTorch, NumPy, SpikingJelly

# This is the definition of relative tool functions and classes.

import sys
import random
import scipy
import torch
import time
import torch
import numpy as np
from spikingjelly.activation_based import neuron, layer
from typing import *
import ast

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
        1. index: list
        2. findex: txt file name
        2. memory_num = 1
        3. channel_amount = 32
        4. mode = 'spiking' (or prep, origin, BSA, zzh)
        5. d_label (download label name) = "basic"
        6. time (seconds, then the timestep will be time*128) = None (select all)
        '''
        from_argv = lambda key, default: argv[key] if key in argv else default
        self.argv=argv
        self.channel_amount = from_argv("channel_amount", 32)
        self.time = from_argv("time", None)
        self.dir=deap_dir
        self.should_shuffle=False
        self.memory_num = from_argv("memory_num", 1)
        self.mode = from_argv("mode", "spiking")
        if self.mode == 'zzh' and self.time != None:
            raise ValueError("Signals in zzh mode cannot be divided.")

        if "findex" in argv:
            fname=argv["findex"]
            with open(fname, 'r') as file:
                index_str=file.read()
                self.index=ast.literal_eval(index_str)
        elif "index" in argv:
            self.index=argv["index"]
        else:
            file_num = 32
            data_num_in_each_file = 40
            if self.mode == "zzh":
                file_num = 1280
                data_num_in_each_file = 10
            self.index=[] 
            for i in range(file_num):
                for j in range(data_num_in_each_file):
                    if self.time is None:
                        self.index.append((i,j))
                    else:
                        for k in range(7680-self.time*128):
                            self.index.append((i,j,k)) # The ith person's jth test, cutted from k
        random.shuffle(self.index)

        if "d_label" in argv:
            self.d_label=argv["d_label"]
        elif "findex" in argv:
            if argv["findex"]=="index_test.txt":
                self.d_label="test"
            elif argv["findex"]=="index_train.txt":
                self.d_label="train"
            elif argv["findex"]=="index_valid.txt":
                self.d_label="valid"
            else:
                self.d_label="basic"
        else:
            self.d_label="basic"
        with open("index_{}.txt".format(self.d_label), 'w') as file:
            file.write(str(self.index))

        self.memory=dict()


    def split(self, test_ratio=0.1):
        '''
        Split the dataset into training, validation and testing.
        '''
        test_len=int(len(self)*test_ratio)
        train_len=len(self)-2*test_len
        sub_SimpleDeap = lambda index_list, d_label_str: SimpleDeap(self.dir, **self.argv, index=index_list, d_label=d_label_str)
        train_data=sub_SimpleDeap(self.index[:train_len], "train")
        valid_data=sub_SimpleDeap(self.index[train_len:train_len+test_len], "valid")
        test_data=sub_SimpleDeap(self.index[train_len+test_len:], "test")
        return train_data, valid_data, test_data


    def __len__(self):
        return len(self.index)


    def __getitem__(self, i):
        if self.should_shuffle==True:
            random.shuffle(self.index)
            self.should_shuffle=False

        if self.time is None:
            person, test = self.index[i]
        else:
            person, test, start = self.index[i]

        if person in self.memory:
            mat=self.memory[person]
        else:
            if len(self.memory.keys()) >= self.memory_num:
                first_key=list(self.memory.keys())[0]
                del self.memory[first_key]
            if self.mode=="BSA" or self.mode=="zzh":
                mat=np.load("{}/signals/{}_signal_{}.npy".format(self.dir, self.mode, person), allow_pickle=True)
            else:
                file_index="0{}".format(person+1) if person<9 else str(person+1)
                mat=scipy.io.loadmat("{}/s{}.mat".format(self.dir, file_index))
            self.memory[person]=mat

        if self.mode=="BSA" or self.mode=="zzh":
            x=mat[test]
        else:
            x=mat['data'][test]
        if self.time is None:
            x = x[:self.channel_amount]
        else:
            x=x[:self.channel_amount, start:start+self.time*128]
        if isinstance(x, torch.Tensor) is False:
            x=torch.tensor(x, dtype=torch.float32)

        if self.mode=="BSA" or self.mode=="zzh":
            mat=np.load("{}/labels/{}_labels_{}.npy".format(self.dir, self.mode, person), allow_pickle=True)
            y=mat[test]
        else:
            y=mat['labels'][test][:2]
        if isinstance(y, torch.Tensor) is False:
            y=torch.tensor(y, dtype=torch.float32)

        if self.mode=="spiking":
            x[x>0]=1.
            x[x<=0]=0.
        elif self.mode=="prep":
            x=x-x.min()
            x=x/x.max()
        else:
            x=x

        if self.mode!="BSA" and self.mode!="zzh":
            y[y<5]=0.
            y[y>=5]=1.

        x=x.to(dtype=torch.float32).view(self.channel_amount, -1)
        y=y.to(dtype=torch.float32)

        return x,y

