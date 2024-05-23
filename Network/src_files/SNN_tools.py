import sys
import torch
import time
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
    sys.stdout.flush()

class Logger():
    '''
    This is made for the record of training process.
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
        with open(self.dir+'/log', 'a') as file:
            timestamp=str(time.time())
            file.write(timestamp+'\n\n')
            modellog=str(self.exe.model)
            file.write(modellog+'\n\n')
            train_length=str(len(self.exe.train_data))
            file.write(train_length+'\n\n')
            file.write(str(self.exe.T)+'\n\n')
            file.write(str(self.exe.Cl_list))
            file.write('\n\n\n')
    def __getitem__(self, index)  -> List[str]:
        with open(self.dir+'/log', 'r') as file:
            content=file.read().split('\n\n\n')
            if index >=0:
                logline=content[-(index+2)]
            else:
                logline=content[-index-1]
            splitted=logline.split('\n\n')
        return splitted
