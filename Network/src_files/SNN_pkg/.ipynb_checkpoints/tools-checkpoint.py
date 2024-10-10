# Xiangnan Zhang 2024, School of Future Technologies, Beijing Institute of Technology
# modified: 2024.10.1
# Dependencies: PyTorch, NumPy, SpikingJelly, MatPlotLib

# This is the definition of relative tool functions and classes.

import sys
import random
import scipy
import torch
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
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
            self.index = np.load(fname).to_list()
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
        ndarr = np.array(self.index)
        np.save("index_{}".format(self.d_label), ndarr)

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
            x=x/x.max()

        if self.mode!="BSA" and self.mode!="zzh":
            y[y<5]=0.
            y[y>=5]=1.

        x=x.to(dtype=torch.float32).view(self.channel_amount, -1)
        y=y.to(dtype=torch.float32)

        return x,y



def test_distribution(exe_model, test_dataloader: torch.utils.data.DataLoader, batch_num = 1) -> matplotlib.figure:
    exe_model.load()
    pred = 0
    ans = 0
    for i,(data,labels) in enumerate(test_dataloader):
        data = data.cuda() if torch.cuda.is_available() else data
        pred = exe_model(data)
        ans = labels.cuda() if torch.cuda.is_available() else labels
        if i+1 == batch_num:
            break
    correct = pred[(pred==ans).all(dim=1)].cpu()
    print(correct)
    class_list = [(0,0), (0,1), (1,0), (1,1)]
    correct_num = []
    total_num = 0
    for class_tup in class_list:
        class_tensor = torch.tensor(list(class_tup), dtype = torch.float32)
        eq_tensor = torch.eq(correct, class_tensor)
        bool_idx = eq_tensor.all(dim=1)
        correct_in_class = correct[bool_idx]
        correct_num.append(len(correct_in_class))
        total_num += len(correct_in_class)
    correct_rates = np.array(correct_num)/total_num
    print("Correct Answer Distribution: {}".format(correct_rates.tolist()))
    fig = plt.figure()
    fig_label_list = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
    plt.bar(fig_label_list, correct_rates)
    plt.title("Correct Answer Distribution ({}% for total)".format(len(correct)/len(ans)*100))
    plt.tight_layout()
    return fig



def balance_data(data: torch.Tensor, labels: torch.Tensor, shuffle = False) -> torch.Tensor:
    '''
    This function is used to balance the amount of data for each class.
    Parameters:
        data, labels: a batch of data and labels. dim=2.
        shuffle: wheter shuffle the balanced data and labels or not. It's not necessary for training.
    '''
    target_list = [[0,0], [1,0], [0,1], [1,1]]
    split_list = []
    minlen = 99999
    for target_origin in target_list:
        target = torch.tensor(target_origin, dtype = torch.float32)
        if torch.cuda.is_available():
            target = target.cuda()
        fit_list = (labels==target).all(dim=1)

        data_fit = data[fit_list]
        labels_fit = labels[fit_list]
        if minlen > len(labels_fit):
            minlen = len(labels_fit)

        split_list.append((data_fit, labels_fit))

    data_list = []
    labels_list = []
    for data_fit, labels_fit in split_list:
        data_fit = data_fit[:minlen]
        labels_fit = labels_fit[:minlen]
        data_list.append(data_fit)
        labels_list.append(labels_fit)

    balanced_data = torch.cat(tuple(data_list), dim=0)
    balanced_labels = torch.cat(tuple(labels_list), dim=0)

    if shuffle:
        indices = torch.randperm(minlen)
        balanced_data = balanced_data[indices]
        balanced_labels = balanced_labels[indices]
    
    return balanced_data, balanced_labels
