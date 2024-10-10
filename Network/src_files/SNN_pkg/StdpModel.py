# By Xiangnan Zhang, School of Future Technologies, Beijing Institute of Technology
# This is the definition of the STDP-based model in Affective Computing Program
# Dependency: SpikingJelly, PyTorch, NumPy, SciPy, MatPlotLib
# Modified: 2024.7.24

from spikingjelly.activation_based import neuron,layer,functional,learning
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor,ToPILImage
import numpy as np
from scipy import ndimage
from typing import *
import matplotlib.pyplot as plt
import time
import ast

from .tools import process_print, Logger

class DOG():  # Different of Gossian
    def __init__(self, sigma1, sigma2, radius):
        self.sigma1=sigma1
        self.sigma2=sigma2
        self.radius=radius
    def __call__(self,x: torch.Tensor) -> torch.Tensor:
        x_arr=x.detach().cpu().numpy()
        blurred1=ndimage.gaussian_filter(x_arr, self.sigma1, radius=self.radius)
        blurred2=ndimage.gaussian_filter(x_arr, self.sigma2, radius=self.radius)
        dog_res=np.abs(blurred1-blurred2)
        res_tensor=torch.from_numpy(dog_res)
        if torch.cuda.is_available() == True:
            res_tensor=res_tensor.cuda()
        else:
            res_tensor.cpu()
        return res_tensor

class WTA_LIFNode(neuron.LIFNode):
    '''
    LIF node that obeys winner-takes-all mechanism.
    '''
    pre_v=0
    def neuronal_fire(self):
        can_firing_arg = self.v.argmax(1)
        mask=torch.zeros_like(self.v, dtype=torch.float32)
        mask = mask.cuda() if torch.cuda.is_available() else mask
        for i,arg in enumerate(can_firing_arg):
            mask[i, arg.item()] = 1.
        WTA_v = torch.mul(self.v, mask)
        return self.surrogate_function(WTA_v - self.v_threshold)
    def neuronal_reset(self, spike):
        self.pre_v=self.v
        spike=self.surrogate_function(self.v - self.v_threshold)
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike
        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)
        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

class CV_STDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dog=DOG(2,0,3)  # Kernel size: 7*7
        self.node1=neuron.IFNode(4.5)
        self.conv1=layer.Conv2d(1,4,3,padding=0, bias=False)  #out: 26*26
        self.node2=neuron.IFNode(4.5)
        self.pool1=layer.MaxPool2d(2)  #out: 13*13
        self.conv2=layer.Conv2d(4,4,3,padding=0, bias=False)  #out: 11*11
        self.node3=neuron.IFNode(1.)
        self.pool2=layer.MaxPool2d(2, padding=1) #out: 6*6
        self.conv3=layer.Conv2d(4,1,3,padding=0, bias=False)  #out: 4*4
        self.node4=neuron.LIFNode(2.)
        self.record=False
        self.log=[[],[],[],[]]
        self.rates=[]
        for param in self.parameters():
            param.data=torch.abs(param.data)
    def __getitem__(self, index):
        '''
        It should match the protocol of STDPExe class.
        A synapse and a neuron should be a pair.
        '''
        lis=[[self.conv1, self.node2], [self.conv2, self.node3], [self.conv3, self.node4]]
        return lis[index]
    def __len__(self):
        return 3
    def __single_forward(self, x: torch.Tensor, record_rate: bool = False) -> torch.Tensor:
        '''
        A single propogation process without time steps.
        x: torch.Tensor, [n, 1, 28, 28]
        return: torch.Tensor, [n, 4, 4]
        '''
        x=self.dog(x)
        x/=x.max()
        x=self.node1(x)
        if self.record==True:
            self.log[0]=x
        block=[
        [self.conv1, self.node2, self.pool1],
        [self.conv2, self.node3, self.pool2],
        [self.conv3, self.node4],
        ]
        if record_rate==True:
            rate_list=[]
        for i,sequence in enumerate(block):
            for lay in sequence:
                x=lay(x)  # Shape: [batch_size, out_channels, row, col]
                if record_rate==True and isinstance(lay, neuron.BaseNode):
                    spiking=x[0].flatten()
                    rate=spiking.sum()/len(spiking)
                    rate_list.append(rate.item())
            if self.record==True:
                self.log[i+1]=x
        if record_rate==True:
            self.rates=rate_list
        return x.view(-1,4,4)
    def forward(self, x: torch.Tensor, record_rate: bool = False) -> torch.Tensor:
        with torch.no_grad():
            x=x.view(-1,1,28,28)
        functional.reset_net(self)
        pred=0
        T=30
        self.record=False
        do_record_rate=False
        for i in range(T):
            if i==T-1:
                self.record=True
                if record_rate==True:
                    do_record_rate=True
            pred+=self.__single_forward(x, do_record_rate)
        return (pred/T).view(-1, 4, 4)

class EEG_LSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1_linear=layer.Linear(32,64,bias=False)
        self.b1_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b1_cache=0
        self.b2_linear=layer.Linear(64,128,bias=False)
        self.b2_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b2_cache=0
        self.b2_recurrent=layer.Linear(128,128,bias=False)
        self.b3_linear=layer.Linear(128,128,bias=False)
        self.b3_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b3_cache=0
        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_linear, self.b1_neuron],
            [self.b2_linear, self.b2_neuron],
            [self.b2_recurrent, self.b2_neuron],
            [self.b3_linear, self.b3_neuron]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.65, std=0.3)
            param.data=torch.clamp(param.data, min=0, max=1)
    def __getitem__(self, index):
        return self.synapse_list[index]
    def __len__(self):
        return len(self.synapse_list)
    def forward(self, x, record_rate: bool = False, auto_reset: bool = True):
        '''
        x: shape [batch_size, 32, time_step]
        out: [batch_size, 500]
        '''
        batch_size,l,time_step=x.shape
        self.b1_cache=torch.zeros(batch_size,64)
        self.b2_cache=torch.zeros(batch_size,128)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
            self.b2_cache=self.b2_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            inp=x[:,:,i]  # shape: [batch_size, 32]
            inp=self.b1_linear(inp)
            inp=self.b1_neuron(inp)
            self.b1_cache=inp
            inp=self.b2_linear(inp)
            y_cache=self.b2_recurrent(self.b2_cache)
            inp=self.b2_neuron(inp+y_cache)
            self.b2_cache=inp
            inp=self.b3_linear(inp)
            inp=self.b3_neuron(inp)
            self.b3_cache=inp
            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                n_1=self.b1_cache[0].detach()
                n_2=self.b2_cache[0].detach()
                n_3=self.b3_cache[0].detach()
                rate_list.append((n_1.sum()/len(n_1)).cpu().item())
                rate_list.append((n_2.sum()/len(n_2)).cpu().item())
                rate_list.append((n_3.sum()/len(n_3)).cpu().item())
                self.rates=rate_list
        liquid_state=torch.exp(self.b3_neuron.pre_v)
        return liquid_state

class EEG_SimpleLSM(nn.Module):
    def __init__(self):
        super().__init__()
        self.trans_neuron = WTA_LIFNode(tau=3., v_threshold=1.5, decay_input=False)
        self.b1_linear=layer.Linear(32,64,bias=False)
        self.b1_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b1_cache=0
        self.b2_linear=layer.Linear(64,128,bias=False)
        self.b2_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b2_cache=0
        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_linear, self.b1_neuron],
            [self.b2_linear, self.b2_neuron]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.8, std=0.05)
            param.data=torch.clamp(param.data, min=0, max=1)
    def __getitem__(self, index):
        return self.synapse_list[index]
    def __len__(self):
        return len(self.synapse_list)
    def forward(self, x, record_rate: bool = False, auto_reset: bool = True):
        '''
        x: shape [batch_size, 32, time_step]
        out: [batch_size, 500]
        '''
        batch_size,l,time_step=x.shape
        self.b1_cache=torch.zeros(batch_size,64)
        self.b2_cache=torch.zeros(batch_size,128)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
            self.b2_cache=self.b2_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            inp=x[:,:,i]  # shape: [batch_size, 32]
            inp=self.trans_neuron(inp)
            inp=self.b1_linear(inp)
            inp=self.b1_neuron(inp)
            self.b1_cache=inp
            inp=self.b2_linear(inp)
            inp=self.b2_neuron(inp)
            self.b2_cache=inp
            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                n_1=self.b1_cache[0].detach()
                n_2=self.b2_cache[0].detach()
                rate_list.append((n_1.sum()/len(n_1)).cpu().item())
                rate_list.append((n_2.sum()/len(n_2)).cpu().item())
                self.rates=rate_list
        liquid_state=torch.exp(self.b2_neuron.pre_v)
        return liquid_state

class EEG_Double(nn.Module):
    def __init__(self, decrease_factor = 0.5):
        self.decrease_factor=decrease_factor
        super().__init__()
        self.b1_linear=layer.Linear(14,14,bias=False)
        self.b1_lif=neuron.LIFNode(v_threshold=7.)
        self.b1_cache=0
        self.b2_linear=layer.Linear(14,500,bias=False)
        self.b2_lif=neuron.LIFNode(v_threshold=7.)
        self.b2_cache=0
        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_linear, self.b1_lif],
            [self.b2_linear, self.b2_lif]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.8, std=0.05)
            param.data=torch.clamp(param.data, min=0, max=1)
    def __getitem__(self, index):
        return self.synapse_list[index]
    def __len__(self):
        return len(self.synapse_list)
    def forward(self, x, record_rate: bool = False, auto_reset: bool = True):
        '''
        x: shape [batch_size, 14, time_step]
        out: [batch_size, 500]
        '''
        batch_size,l,time_step=x.shape
        self.b1_cache=torch.zeros(batch_size,14)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            with torch.no_grad():
                inp=x[:,:,i]  # shape: [batch_size, 14]
            inp=self.b1_linear(inp)
            inp+=self.decrease_factor*self.b1_cache  # recurrent structure
            inp=self.b1_lif(inp)
            self.b1_cache=inp
            inp=self.b2_linear(inp)
            inp=self.b2_lif(inp)
            self.b2_cache=inp
            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                n_1=self.b1_cache[0].detach()
                n_2=self.b2_cache[0].detach()
                rate_list.append((n_1.sum()/len(n_1)).cpu().item())
                rate_list.append((n_2.sum()/len(n_2)).cpu().item())
                self.rates=rate_list
        liquid_state=torch.exp(self.b2_lif.v)
        return liquid_state

class EEG_SequentialCompressionUnit(nn.Module):
    def __init__(self, mode="spiking", **argv):
        '''
        Parameters in argv:
        1. channel amount: default 32
        2. mode: spiking or realnum, default spiking
        3. WTA: when treating the recurrent process, whether use winner-takes-all mechanism. Default True.
        '''
        super().__init__()
        # Following: Parameter Analysis
        from_argv = lambda key,default: argv[key] if key in argv else default
        self.channel_amount = from_argv("channel_amount", 32)
        self.mode = from_argv("mode", "spiking")
        self.WTA = from_argv("WTA", True)
        # Following: Nodes in Calculation Chart
        self.cache=0
        self.forward_linear=layer.Linear(self.channel_amount, self.channel_amount, bias=False)
        self.recurrent_linear=layer.Linear(self.channel_amount, self.channel_amount, bias=False)
        if self.mode=="spiking":
            if self.WTA==True:
                self.neuron = WTA_LIFNode(tau=6.0, v_threshold=3.0, decay_input=False)
            else:
                self.neuron=neuron.LIFNode(tau=10.0, v_threshold=3.0, decay_input=False)
        else:
            self.neuron=neuron.LIFNode(tau=6.0, v_threshold=3.0, decay_input=False)
        self.synapse_list=[
        [self.forward_linear, self.neuron],
        [self.recurrent_linear, self.neuron]
        ]
        # Following: Model Initialization
        if self.mode=="spiking" and self.WTA is False:
            init_mean=0.2
            init_std=0.08
        else:
            init_mean=0.8
            init_std=0.2
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=init_mean, std=init_std)
            param.data=torch.clamp(param.data, min=0, max=1)
        # Following: Port
        self.rates=None
    def __getitem__(self, index):
        return self.synapse_list[index]
    def __len__(self):
        return len(self.synapse_list)
    def forward(self, x, record_rate: bool = False, auto_reset: bool = True):
        '''
        x: shape [batch_size, channel_amount, time_step]
        out: [batch_size, channel_amount]
        '''
        batch_size,l,time_step=x.shape
        self.cache=torch.zeros(batch_size,self.channel_amount)
        if torch.cuda.is_available()==True:
            self.cache=self.cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            with torch.no_grad():
                inp=x[:,:,i]  # shape: [batch_size, channel_amount]
                inp=self.forward_linear(inp)
            if self.WTA is True:
                inp=inp+self.recurrent_linear(self.cache)
            else:
                inp=inp+2*self.recurrent_linear(self.cache)/self.channel_amount
            inp=self.neuron(inp)
            self.cache=inp
        if record_rate==True:
            # Then record the voltage reset rates of each neuron layer.
            rate_list=[]
            spiking=self.neuron.v[0].detach()
            spiking=(spiking<=1e-5).int().float()
            rate_list.append((spiking.sum()/len(spiking)).cpu().item())
            self.rates=rate_list
        liquid_state=torch.exp(self.neuron.pre_v)
        return liquid_state


class EegExposeAll_32_m(nn.Module):
    '''
    For EEG. Expose the votage of all the neurons. Use v as the output rather than pre_v.
    It is aimed to let the classifier fully connect to this LSM. While the structure is the same as EEG_LSM.
    in: [n,32]
    out: tuple, ([n, 64], [n, 128], [n, 128])
    Only compatible with the fully connected classifier.
    Written date: 2024.9.22
    '''
    def __init__(self):
        super().__init__()
        self.b1_linear=layer.Linear(32,64,bias=False)
        self.b1_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b1_cache=0
        self.b2_linear=layer.Linear(64,128,bias=False)
        self.b2_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b2_cache=0
        self.b2_recurrent=layer.Linear(128,128,bias=False)
        self.b3_linear=layer.Linear(128,128,bias=False)
        self.b3_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b3_cache=0
        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_linear, self.b1_neuron],
            [self.b2_linear, self.b2_neuron],
            [self.b2_recurrent, self.b2_neuron],
            [self.b3_linear, self.b3_neuron]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.65, std=0.3)
            param.data=torch.clamp(param.data, min=0, max=1)
    def __getitem__(self, index):
        return self.synapse_list[index]
    def __len__(self):
        return len(self.synapse_list)
    def forward(self, x, record_rate: bool = False, auto_reset: bool = True) -> Tuple:
        '''
        x: shape [batch_size, 32, time_step]
        out: [batch_size, 500]
        '''
        batch_size,l,time_step=x.shape
        self.b1_cache=torch.zeros(batch_size,64)
        self.b2_cache=torch.zeros(batch_size,128)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
            self.b2_cache=self.b2_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            inp=x[:,:,i]  # shape: [batch_size, 32]
            inp=self.b1_linear(inp)
            inp=self.b1_neuron(inp)
            self.b1_cache=inp
            inp=self.b2_linear(inp)
            y_cache=self.b2_recurrent(self.b2_cache)
            inp=self.b2_neuron(inp+y_cache)
            self.b2_cache=inp
            inp=self.b3_linear(inp)
            inp=self.b3_neuron(inp)
            self.b3_cache=inp
            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                n_1=self.b1_cache[0].detach()
                n_2=self.b2_cache[0].detach()
                n_3=self.b3_cache[0].detach()
                rate_list.append((n_1.sum()/len(n_1)).cpu().item())
                rate_list.append((n_2.sum()/len(n_2)).cpu().item())
                rate_list.append((n_3.sum()/len(n_3)).cpu().item())
                self.rates=rate_list
        b1_state = torch.exp(self.b1_neuron.v)
        b2_state = torch.exp(self.b2_neuron.v)
        b3_state = torch.exp(self.b3_neuron.v)
        return b1_state, b2_state, b3_state
# end of EegExposeAll_32_m



class EegExposeTwo_14_m(nn.Module):
    '''
    For EEG. Expose the votage of last 2 layers of the neurons. Use v as the output rather than pre_v.
    It is aimed to let the classifier fully connect to this LSM. While the structure is the same as EEG_LSM.
    in: [n,14]
    out: tuple, ([n, 128], [n, 128])
    Only compatible with the fully connected classifier.
    Written date: 2024.10.4
    '''
    def __init__(self, use_pre_v = False):
        super().__init__()
        self.use_pre_v = use_pre_v

        self.b1_linear=layer.Linear(14,64,bias=False)
        self.b1_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b1_cache=0
        self.b2_linear=layer.Linear(64,128,bias=False)
        self.b2_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b2_cache=0
        self.b2_recurrent=layer.Linear(128,128,bias=False)
        self.b3_linear=layer.Linear(128,128,bias=False)
        self.b3_neuron=WTA_LIFNode(tau=80000., v_threshold=1.2, decay_input=False)
        self.b3_cache=0
        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_linear, self.b1_neuron],
            [self.b2_linear, self.b2_neuron],
            [self.b2_recurrent, self.b2_neuron],
            [self.b3_linear, self.b3_neuron]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.65, std=0.3)
            param.data=torch.clamp(param.data, min=0, max=1)
    def __getitem__(self, index):
        return self.synapse_list[index]
    def __len__(self):
        return len(self.synapse_list)
    def forward(self, x, record_rate: bool = False, auto_reset: bool = True) -> Tuple:
        batch_size,l,time_step=x.shape
        self.b1_cache=torch.zeros(batch_size,64)
        self.b2_cache=torch.zeros(batch_size,128)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
            self.b2_cache=self.b2_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            inp=x[:,:,i]  # shape: [batch_size, 32]
            inp=self.b1_linear(inp)
            inp=self.b1_neuron(inp)
            self.b1_cache=inp
            inp=self.b2_linear(inp)
            y_cache=self.b2_recurrent(self.b2_cache)
            inp=self.b2_neuron(inp+y_cache)
            self.b2_cache=inp
            inp=self.b3_linear(inp)
            inp=self.b3_neuron(inp)
            self.b3_cache=inp
            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                n_1=self.b1_cache[0].detach()
                n_2=self.b2_cache[0].detach()
                n_3=self.b3_cache[0].detach()
                rate_list.append((n_1.sum()/len(n_1)).cpu().item())
                rate_list.append((n_2.sum()/len(n_2)).cpu().item())
                rate_list.append((n_3.sum()/len(n_3)).cpu().item())
                self.rates=rate_list
        if self.use_pre_v:
            b2_state = torch.exp(self.b2_neuron.pre_v)
            b3_state = torch.exp(self.b3_neuron.pre_v)
        else:
            b2_state = torch.exp(self.b2_neuron.v)
            b3_state = torch.exp(self.b3_neuron.v)
        return b2_state, b3_state
# end of EegExpose_14_m



class EegCircle_14_m(nn.Module):
    '''
    Use a recurrent connection between the 2rd and 3th neuron layer.
    in: [n,14]
    out: tuple, ([n, 128], [n, 128])
    Only compatible with the fully connected classifier.
    Written date: 2024.10.4
    '''
    def __init__(self, use_pre_v = False):
        super().__init__()
        self.use_pre_v = use_pre_v

        self.b1_linear=layer.Linear(14,64,bias=False)
        self.b1_neuron=WTA_LIFNode(tau=800., v_threshold=1.2, decay_input=False)
        self.b1_cache=0
        self.b2_linear=layer.Linear(64,128,bias=False)
        self.b2_neuron=WTA_LIFNode(tau=800., v_threshold=1.2, decay_input=False)
        self.b2_cache=0
        self.recurrent=layer.Linear(128,128,bias=False)
        self.b3_linear=layer.Linear(128,128,bias=False)
        self.b3_neuron=WTA_LIFNode(tau=800., v_threshold=1.2, decay_input=False)
        self.b3_cache=0
        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_linear, self.b1_neuron],
            [self.b2_linear, self.b2_neuron],
            [self.b3_linear, self.b3_neuron],
            [self.recurrent, self.b2_neuron]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.65, std=0.3)
            param.data=torch.clamp(param.data, min=0, max=1)
    def __getitem__(self, index):
        return self.synapse_list[index]
    def __len__(self):
        return len(self.synapse_list)
    def forward(self, x, record_rate: bool = False, auto_reset: bool = True) -> Tuple:
        batch_size,l,time_step=x.shape
        self.b1_cache=torch.zeros(batch_size,64)
        self.b2_cache=torch.zeros(batch_size,128)
        self.b3_cache=torch.zeros(batch_size, 128)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
            self.b2_cache=self.b2_cache.cuda()
            self.b3_cache = self.b3_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            inp=x[:,:,i]  # shape: [batch_size, 14]
            inp=self.b1_linear(inp)
            inp=self.b1_neuron(inp)
            self.b1_cache=inp
            inp=self.b2_linear(inp)
            y_cache=self.recurrent(self.b3_cache)
            inp=self.b2_neuron(inp+y_cache)
            self.b2_cache=inp
            inp=self.b3_linear(inp)
            inp=self.b3_neuron(inp)
            self.b3_cache=inp
            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                n_1=self.b1_cache[0].detach()
                n_2=self.b2_cache[0].detach()
                n_3=self.b3_cache[0].detach()
                rate_list.append((n_1.sum()/len(n_1)).cpu().item())
                rate_list.append((n_2.sum()/len(n_2)).cpu().item())
                rate_list.append((n_3.sum()/len(n_3)).cpu().item())
                self.rates=rate_list
        if self.use_pre_v:
            b2_state = torch.exp(self.b2_neuron.pre_v)
            b3_state = torch.exp(self.b3_neuron.pre_v)
        else:
            b2_state = torch.exp(self.b2_neuron.v)
            b3_state = torch.exp(self.b3_neuron.v)
        return b2_state, b3_state
# end of EegCircle_14_m



class EegLeaky_14_m256(nn.Module):
    '''
    Use a recurrent connection between the 2rd and 3th neuron layer, and with a small tau for last two layer.
    i.e. with Leaky LIF.
    Leaky makes robustness.
    Also with a larger scale.
    in: [n,14]
    out: tuple, ([n, 256], [n, 256])
    Only compatible with the fully connected classifier.
    Written date: 2024.10.5
    '''
    def __init__(self, use_pre_v = False):
        super().__init__()
        self.use_pre_v = use_pre_v

        self.b1_linear=layer.Linear(14,64,bias=False)
        self.b1_neuron=WTA_LIFNode(tau=800., v_threshold=1.2, decay_input=False)
        self.b1_cache=0
        self.b2_linear=layer.Linear(64,256,bias=False)
        self.b2_neuron=WTA_LIFNode(tau=3., v_threshold=1.2, decay_input=False)
        self.b2_cache=0
        self.recurrent=layer.Linear(256,256,bias=False)
        self.b3_linear=layer.Linear(256,256,bias=False)
        self.b3_neuron=WTA_LIFNode(tau=3., v_threshold=1.2, decay_input=False)
        self.b3_cache=0
        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_linear, self.b1_neuron],
            [self.b2_linear, self.b2_neuron],
            [self.b3_linear, self.b3_neuron],
            [self.recurrent, self.b2_neuron]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.65, std=0.3)
            param.data=torch.clamp(param.data, min=0, max=1)
    def __getitem__(self, index):
        return self.synapse_list[index]
    def __len__(self):
        return len(self.synapse_list)
    def forward(self, x, record_rate: bool = False, auto_reset: bool = True) -> Tuple:
        batch_size,l,time_step=x.shape
        self.b1_cache=torch.zeros(batch_size,64)
        self.b2_cache=torch.zeros(batch_size,256)
        self.b3_cache=torch.zeros(batch_size, 256)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
            self.b2_cache=self.b2_cache.cuda()
            self.b3_cache = self.b3_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            inp=x[:,:,i]  # shape: [batch_size, 14]
            inp=self.b1_linear(inp)
            inp=self.b1_neuron(inp)
            self.b1_cache=inp
            inp=self.b2_linear(inp)
            y_cache=self.recurrent(self.b3_cache)
            inp=self.b2_neuron(inp+y_cache)
            self.b2_cache=inp
            inp=self.b3_linear(inp)
            inp=self.b3_neuron(inp)
            self.b3_cache=inp
            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                n_1=self.b1_cache[0].detach()
                n_2=self.b2_cache[0].detach()
                n_3=self.b3_cache[0].detach()
                rate_list.append((n_1.sum()/len(n_1)).cpu().item())
                rate_list.append((n_2.sum()/len(n_2)).cpu().item())
                rate_list.append((n_3.sum()/len(n_3)).cpu().item())
                self.rates=rate_list
        if self.use_pre_v:
            b2_state = torch.exp(self.b2_neuron.pre_v)
            b3_state = torch.exp(self.b3_neuron.pre_v)
        else:
            b2_state = torch.exp(self.b2_neuron.v)
            b3_state = torch.exp(self.b3_neuron.v)
        return b2_state, b3_state
# end of EegLeaky_14_m256



class EegFunctionColumn_14_5m128(nn.Module):
    '''
    This class aims to simulate 
    in: [n,14]
    out: tuple, ([n, 128], [n, 128])
    Only compatible with the fully connected classifier.
    Written date: 2024.10.5
    '''
    def __init__(self):
        super().__init__()

        self.b1_bridge=layer.Linear(14,64,bias=False)
        self.b1_inside = layer.Linear(64, 64, bias = False)
        self.b1_neuron = WTA_LIFNode(tau=3., v_threshold=1.2, decay_input=False)
        self.b1_cache=0

        self.recurrent_2_1 = layer.Linear(128, 64, bias = False)

        self.b2_bridge = layer.Linear(64,128,bias=False)
        self.b2_inside = layer.Linear(128, 128, bias=False)
        self.b2_neuron=WTA_LIFNode(tau=3., v_threshold=1.2, decay_input=False)
        self.b2_cache=0

        self.recurrent_3_2 = layer.Linear(128,128,bias=False)

        self.b3_bridge = layer.Linear(128,128,bias=False)
        self.b3_inside = layer.Linear(128, 128, bias = False)
        self.b3_neuron = WTA_LIFNode(tau=3., v_threshold=1.2, decay_input=False)
        self.b3_cache=0

        self.recurrent_4_3 = layer.Linear(128,128,bias=False)

        self.b4_bridge = layer.Linear(128,128,bias=False)
        self.b4_inside = layer.Linear(128, 128, bias = False)
        self.b4_neuron = WTA_LIFNode(tau=3., v_threshold=1.2, decay_input=False)
        self.b4_cache=0

        self.recurrent_5_4 = layer.Linear(128,128,bias=False)

        self.b5_bridge = layer.Linear(128,128,bias=False)
        self.b5_inside = layer.Linear(128, 128, bias = False)
        self.b5_neuron = WTA_LIFNode(tau=3., v_threshold=1.2, decay_input=False)
        self.b5_cache=0

        self.recurrent_6_5 = layer.Linear(128,128,bias=False)

        self.b6_bridge = layer.Linear(128,128,bias=False)
        self.b6_inside = layer.Linear(128, 128, bias = False)
        self.b6_neuron = WTA_LIFNode(tau=3., v_threshold=1.2, decay_input=False)
        self.b6_cache=0

        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_bridge, self.b1_neuron],
            [self.b1_inside, self.b1_neuron],
            [self.b2_bridge, self.b2_neuron],
            [self.b2_inside, self.b2_neuron],
            [self.b3_bridge, self.b3_neuron],
            [self.b3_inside, self.b3_neuron],
            [self.b4_bridge, self.b4_neuron],
            [self.b4_inside, self.b4_neuron],
            [self.b5_bridge, self.b5_neuron],
            [self.b5_inside, self.b5_neuron],
            [self.b6_bridge, self.b6_neuron],
            [self.b6_inside, self.b6_neuron],
            [self.recurrent_2_1, self.b1_neuron],
            [self.recurrent_3_2, self.b2_neuron],
            [self.recurrent_4_3, self.b3_neuron],
            [self.recurrent_5_4, self.b4_neuron],
            [self.recurrent_6_5, self.b5_neuron]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.65, std=0.3)
            param.data=torch.clamp(param.data, min=0, max=1)

    def __getitem__(self, index):
        return self.synapse_list[index]

    def __len__(self):
        return len(self.synapse_list)

    def forward(self, x, record_rate: bool = False, auto_reset: bool = True) -> Tuple:
        batch_size,l,time_step=x.shape
        self.b1_cache=torch.zeros(batch_size,64)
        self.b2_cache=torch.zeros(batch_size,128)
        self.b3_cache=torch.zeros(batch_size, 128)
        self.b4_cache=torch.zeros(batch_size, 128)
        self.b5_cache=torch.zeros(batch_size, 128)
        self.b6_cache=torch.zeros(batch_size, 128)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
            self.b2_cache=self.b2_cache.cuda()
            self.b3_cache = self.b3_cache.cuda()
            self.b4_cache=self.b4_cache.cuda()
            self.b5_cache=self.b5_cache.cuda()
            self.b6_cache = self.b6_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)

        for i in range(time_step):
            inp=x[:,:,i]  # shape: [batch_size, 14]

            stalk=self.b1_bridge(inp)
            stalk=self.b1_neuron(stalk)
            inside = self.b1_inside(stalk)
            _ = self.b1_neuron(inside)

            stalk = self.b2_bridge(stalk)
            stalk = self.b2_neuron(stalk)
            inside = self.b2_inside(stalk)
            _ = self.b2_neuron(inside)

            stalk = self.b3_bridge(stalk)
            stalk = self.b3_neuron(stalk)
            inside = self.b3_inside(stalk)
            _ = self.b3_neuron(inside)

            stalk=self.b4_bridge(stalk)
            stalk=self.b4_neuron(stalk)
            inside = self.b4_inside(stalk)
            _ = self.b4_neuron(inside)

            stalk = self.b5_bridge(stalk)
            stalk = self.b5_neuron(stalk)
            inside = self.b5_inside(stalk)
            _ = self.b5_neuron(inside)

            stalk = self.b6_bridge(stalk)
            stalk = self.b6_neuron(stalk)
            inside = self.b6_inside(stalk)
            _ = self.b6_neuron(inside)

            self.b6_cache = _

            stalk = self.recurrent_6_5(stalk)
            stalk = self.b5_neuron(stalk)
            inside = self.b5_inside(stalk)
            _ = self.b5_neuron(inside)

            self.b5_cache = _

            stalk = self.recurrent_5_4(stalk)
            stalk = self.b4_neuron(stalk)
            inside = self.b4_inside(stalk)
            _ = self.b4_neuron(inside)

            self.b4_cache = _

            stalk = self.recurrent_4_3(stalk)
            stalk = self.b3_neuron(stalk)
            inside = self.b3_inside(stalk)
            _ = self.b3_neuron(inside)

            self.b3_cache = _

            stalk = self.recurrent_3_2(stalk)
            stalk = self.b2_neuron(stalk)
            inside = self.b2_inside(stalk)
            _ = self.b2_neuron(inside)

            self.b2_cache = _

            stalk = self.recurrent_2_1(stalk)
            stalk = self.b1_neuron(stalk)
            inside = self.b1_inside(stalk)
            _ = self.b1_neuron(inside)

            self.b1_cache = _

            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                n_1=self.b1_cache[0].detach()
                n_2=self.b2_cache[0].detach()
                n_3=self.b3_cache[0].detach()
                n_4=self.b4_cache[0].detach()
                n_5=self.b5_cache[0].detach()
                n_6=self.b6_cache[0].detach()
                rate_list.append((n_1.sum()/len(n_1)).cpu().item())
                rate_list.append((n_2.sum()/len(n_2)).cpu().item())
                rate_list.append((n_3.sum()/len(n_3)).cpu().item())
                rate_list.append((n_4.sum()/len(n_4)).cpu().item())
                rate_list.append((n_5.sum()/len(n_5)).cpu().item())
                rate_list.append((n_6.sum()/len(n_6)).cpu().item())
                self.rates=rate_list
        b2_state = torch.exp(self.b2_neuron.v)
        b3_state = torch.exp(self.b3_neuron.v)
        b4_state = torch.exp(self.b4_neuron.v)
        b5_state = torch.exp(self.b5_neuron.v)
        b6_state = torch.exp(self.b6_neuron.v)
        return b2_state, b3_state, b4_state, b5_state, b6_state
# end of EegFunctionColumn_14_5m128



class EegFunctionColumnAsyn_14_5m128(nn.Module):
    '''
    This class aims to simulate the ASYNCHRONOUS process happening in Function Column. 
    Use exponential decay of time as output states. See Al Zoubi et al. 2018.
    in: [n,14]
    out: tuple, ([n, 128])*5
    Only compatible with the fully connected classifier.
    Written date: 2024.10.7
    '''
    def __init__(self, **argv):
        '''
        In argv:
            1. decay: default 100
        '''
        self.decay = argv["decay"] if "decay" in argv else 100

        super().__init__()

        co_tau = 5.0
        co_v = 1.2

        self.b1_bridge=layer.Linear(14,64,bias=False)
        self.b1_inside = layer.Linear(64, 64, bias = False)
        self.b1_neuron = WTA_LIFNode(tau=co_tau, v_threshold=co_v, decay_input=False)

        self.recurrent_2_1 = layer.Linear(128, 64, bias = False)

        self.b2_bridge = layer.Linear(64,128,bias=False)
        self.b2_inside = layer.Linear(128, 128, bias=False)
        self.b2_neuron = WTA_LIFNode(tau=co_tau, v_threshold=co_v, decay_input=False)

        self.recurrent_3_2 = layer.Linear(128,128,bias=False)

        self.b3_bridge = layer.Linear(128,128,bias=False)
        self.b3_inside = layer.Linear(128, 128, bias = False)
        self.b3_neuron = WTA_LIFNode(tau=co_tau, v_threshold=co_v, decay_input=False)

        self.recurrent_4_3 = layer.Linear(128,128,bias=False)

        self.b4_bridge = layer.Linear(128,128,bias=False)
        self.b4_inside = layer.Linear(128, 128, bias = False)
        self.b4_neuron = WTA_LIFNode(tau=co_tau, v_threshold=co_v, decay_input=False)

        self.recurrent_5_4 = layer.Linear(128,128,bias=False)

        self.b5_bridge = layer.Linear(128,128,bias=False)
        self.b5_inside = layer.Linear(128, 128, bias = False)
        self.b5_neuron = WTA_LIFNode(tau=co_tau, v_threshold=co_v, decay_input=False)

        self.recurrent_6_5 = layer.Linear(128,128,bias=False)

        self.b6_bridge = layer.Linear(128,128,bias=False)
        self.b6_inside = layer.Linear(128, 128, bias = False)
        self.b6_neuron = WTA_LIFNode(tau=co_tau, v_threshold=co_v, decay_input=False)
        
        self.time_counter = [None for _ in range(5)]  # Last Spiking Time. IMPORTANT!
        self.spike_cache = [None for _ in range(6)]

        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_bridge, self.b1_neuron],
            [self.b1_inside, self.b1_neuron],
            [self.b2_bridge, self.b2_neuron],
            [self.b2_inside, self.b2_neuron],
            [self.b3_bridge, self.b3_neuron],
            [self.b3_inside, self.b3_neuron],
            [self.b4_bridge, self.b4_neuron],
            [self.b4_inside, self.b4_neuron],
            [self.b5_bridge, self.b5_neuron],
            [self.b5_inside, self.b5_neuron],
            [self.b6_bridge, self.b6_neuron],
            [self.b6_inside, self.b6_neuron],
            [self.recurrent_2_1, self.b1_neuron],
            [self.recurrent_3_2, self.b2_neuron],
            [self.recurrent_4_3, self.b3_neuron],
            [self.recurrent_5_4, self.b4_neuron],
            [self.recurrent_6_5, self.b5_neuron]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.65, std=0.3)
            param.data=torch.clamp(param.data, min=0, max=1)

    def __getitem__(self, index):
        return self.synapse_list[index]

    def __len__(self):
        return len(self.synapse_list)

    def forward(self, x, record_rate: bool = False, auto_reset: bool = True) -> Tuple:
        batch_size,l,time_step=x.shape

        for i in range(5):
            self.time_counter[i] = torch.zeros(batch_size, 128)
            if torch.cuda.is_available():
                self.time_counter[i] = self.time_counter[i].cuda()

        for i in range(6):
            self.spike_cache[i] = torch.zeros(batch_size, 128 if i!=0 else 64)
            if torch.cuda.is_available():
                self.spike_cache[i] = self.spike_cache[i].cuda()
        
        if auto_reset==True:
            functional.reset_net(self)

        for i in range(time_step):  # Function Column For 6 Blocks. Use Output Based on Last 5 Blocks.
            inp=x[:,:,i]  # shape: [batch_size, 14]
            out = []

            float_1 = self.b1_bridge(inp)
            float_2 = self.b1_inside(self.spike_cache[0])
            float_3 = self.recurrent_2_1(self.spike_cache[1])
            out.append(self.b1_neuron(float_1+float_2+float_3))

            float_1 = self.b2_bridge(self.spike_cache[0])
            float_2 = self.b2_inside(self.spike_cache[1])
            float_3 = self.recurrent_3_2(self.spike_cache[2])
            out.append(self.b2_neuron(float_1+float_2+float_3))

            float_1 = self.b3_bridge(self.spike_cache[1])
            float_2 = self.b3_inside(self.spike_cache[2])
            float_3 = self.recurrent_4_3(self.spike_cache[3])
            out.append(self.b3_neuron(float_1+float_2+float_3))

            float_1 = self.b4_bridge(self.spike_cache[2])
            float_2 = self.b4_inside(self.spike_cache[3])
            float_3 = self.recurrent_5_4(self.spike_cache[4])
            out.append(self.b4_neuron(float_1+float_2+float_3))

            float_1 = self.b5_bridge(self.spike_cache[3])
            float_2 = self.b5_inside(self.spike_cache[4])
            float_3 = self.recurrent_6_5(self.spike_cache[5])
            out.append(self.b5_neuron(float_1+float_2+float_3))

            float_1 = self.b6_bridge(self.spike_cache[4])
            float_2 = self.b6_inside(self.spike_cache[5])
            out.append(self.b6_neuron(float_1+float_2))

            for i,single_out in enumerate(out):
                self.spike_cache[i] = single_out

            for i in range(5):
                self.time_counter[i] += 1
                self.time_counter[i][self.spike_cache[i+1]==1] = 0

            if record_rate==True and i==time_step-1:
                # Then record the firing rates of each neuron layer.
                rate_list=[]
                for cache in self.spike_cache:
                    n = cache[0].detach()
                    rate_list.append((n.sum()/len(n)).cpu().item())
                self.rates=rate_list

        tau = self.decay  # Decay Factor. Control decay speed.
        states = []
        for time in self.time_counter:
            states.append(torch.exp(-time/tau))   # Exponential Decay. See Al Zoubi et al. 2018, Function (7)
        return tuple(states)
# end of EegFunctionColumnAsyn_14_5m128 



def w_dep_factor(a: float, w: torch.Tensor) ->torch.Tensor:
    '''
    Weight Dependence Factor for STDPLearner. More details
    can be found in Kheradpisheh et al.(2018)
    It should be preprocessed by lambda.
    '''
    return a*w*(1-w)
    
class STDPExe():
    '''
    The following attributes are able to be accessed in public:
    1. dir: if you want to change the saving directory.
    2. train_data
    3. test_data
    4. history_rates
    5 .Cl_list: storing all the Cl values during training process.
    6. dist_list: cotaining paramater distribution.
    The following methods are expected to be called in public:
    1. forward(x)
    2. load(): load the state directory of the STDP model.
    3. calc_Cl(): to calculate the current Cl values.
    4. train()
    5. visualize()
    6. with_record(): load Cl list and history rates in 'dir/STDP_process_record.txt'.
    '''
    def __init__(self, model: nn.Module, store_dir: str, train_data: Iterable[Tuple], test_data: Iterable[Tuple], **argv):
        '''
        store_dir refers to a directory of a folder, where to save training results and the model.
        train_data and test_data: iterable, return an (x, y) tuple. They should be packed as batches.
        In argv:
            1. lr: learning rate, default 0.01
        '''
        self.lr = argv["lr"] if "lr" in argv else 0.01

        self.dir=store_dir
        if torch.cuda.is_available()==True:
            print("STDPExe: Use CUDA")
            self.device=torch.device("cuda")
        else:
            print("STDPExe: Use CPU")
            self.device=torch.device("cpu")
        self.model=model.to(self.device)
        self.train_data=train_data
        self.test_data=test_data
        self.optim=torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0)
        self.pre_wdf=lambda w: w_dep_factor(3e-3,w)   # 'wdf' means weight-dependence factor, see Morrison et al. (2018)
        self.post_wdf=lambda w: w_dep_factor(4e-3,w) 
        self.stdp_learners=[
            learning.STDPLearner('s', self.model[i][0], self.model[i][1], 2, 2, self.pre_wdf, self.post_wdf) for i in range(len(self.model))
            ] 
        self.Cl_list=[] # Used to store all the Cl value during training
        self.logger=Logger(self)
        self.history_rates=[]  # histroy firing rates of each neuron layers during training.
        self.dist_list=[]
    def with_record(self):
        '''
        If you have an existed STDP model and want to train it further,
        you can use this function to load self.Cl_list and self.history_rates
        from file "self.dir/STDP_process_record.txt". 
        '''
        with open("{}/STDP_process_record.txt".format(self.dir), 'r') as file:
            content=file.read().split('\n')[-4:-1]
        get_list = lambda i: ast.literal_eval(
            content[i].split('=')[1]
            )
        self.Cl_list=get_list(0)
        self.history_rates=get_list(1)
        self.dist_list=get_list(2)
        print(
        '''
    STDPExe: record length:
        Cl_list: {}
        history_rates: {}
        dist_list: {}
        '''.format(len(self.Cl_list), len(self.history_rates), len(self.dist_list))
        )
    def forward(self, x: torch.Tensor, training=False) -> torch.Tensor:
        '''
        x: a batch of tensors that will be processed into self.model.
        return: encoding result.
        '''
        if torch.cuda.is_available()==True:
            x=x.cuda()
        for learner in self.stdp_learners:
            learner.reset()  # With the reset-step pair, STDPLearner will not store extra delta_w, even in testing process
        pred=self.model(x, True if training==True else False)
        if training==True:
            self.history_rates.append(self.model.rates)
        for learner in self.stdp_learners:
            learner.step(on_grad=True)
        return pred
    def load(self):
        '''
        Load STDP model's state dictionary from the directory.
        '''
        location='cpu'
        if torch.cuda.is_available()==True:
            location='cuda'
        self.model.load_state_dict(torch.load(self.dir+"/STDPModel.pt", map_location=location, weights_only=True))
        print("STDPExe: STDP model loaded.")
    def calc_Cl(self):
        Cl_list=[]
        for i in range(len(self.model)):
            layer_params=self.model[i][0].parameters()
            cl=0
            length=0
            for param in layer_params:
                param=param.data.flatten()
                length+=len(param)
                cl+=torch.sum(param*(1-param)).cpu().item()
            cl=cl/length
            Cl_list.append(cl)
        return Cl_list
    def calc_distribution(self):
        dist_list=[]
        for i in range(len(self.model)):
            layer_params=self.model[i][0].parameters()
            for param in layer_params:
                param = torch.clamp(param, 0, 1)
                frequency=np.zeros(10)
                param=param.data.flatten()
                for element in param:
                    arg=int(element*10)
                    if arg>10:
                        raise ValueError("Unexpected parameter: {}".format(element))
                    if arg==10:
                        arg-=1
                    frequency[arg]+=1.
                frequency=frequency/frequency.sum()
                dist_list.append(frequency.tolist())
        return dist_list
    def train(self, save=True):
        '''
        save: Bool, wether save model's state dictionary
        After launching, you will get following files under store_dir:
        1. logs (txt files):
            1.1 STDP_log.txt: a log file about basic trainning setting. Updated for each epoch.
            1.2 STDP_process_record.txt: a log file recording training process. Updated during each epoch.
        2. charts (jpg files):
            2.1 cl.jpg: a chart recording the alternation of Cls.
            2.2 firing_rate.jpg: a chart recording the firing rate of each neuron layer. 
            2.3 distribution.jpg: recording parameter distribution.
            2.4 feature.jpg: a graphic recording forward process if it is defined in visualize().
        3. STDPModel.pt: the state directory of the STDP model.
        '''
        print("STDPExe: start training.")
        epoch=0
        not_converged=True
        self.model.train()
        while not_converged:
            print("STDPExe: epoch {} started".format(epoch+1))
            for index,(x,y) in enumerate(self.train_data):
                self.optim.zero_grad()
                fire_rate=self.forward(x, True) # set record_rate=True, record the firing rates.
                self.optim.step()
                with torch.no_grad():
                    self.Cl_list.append(self.calc_Cl())
                    self.dist_list.append(self.calc_distribution())
                process_print(index+1, len(self.train_data))
                B_list=[cl<=2e-4 for cl in self.Cl_list[-1]]
                if (index%5==0 and index!=0) or index==len(self.train_data)-1 or all(B_list):
                    print("")
                    with open(self.dir+'/STDP_process_record.txt', 'w') as file:
                        file.write("time={}\n".format(time.time()))
                        file.write("epoch {}, {}/{}\n".format(epoch+1, index+1, len(self.train_data)))
                        file.write("Cl={}\n".format(self.Cl_list))
                        file.write("spiking_rates={}\n".format(self.history_rates))
                        file.write("distribution={}\n".format(self.dist_list))
                    figs = list(self.visualize(True))
                    for fig_idx in range(len(figs)):
                        plt.close(figs[fig_idx])
                    if save==True:
                        torch.save(self.model.state_dict(), self.dir+"/STDPModel.pt")
                        print("STDPExe: Model saved.")
                if all(B_list):  # Whether all the Cl values are less than 1e-5
                    print("STDPExe: The Model has converged.")
                    not_converged=False
                    break
            self.logger.write_log()
            epoch+=1
        print("STDPExe: training finished.")
    def visualize(self, save=True) -> tuple:
        plt.rcParams['axes.unicode_minus']=False
        '''
        The following charts will be created:
        1. cl_fig (cl.jpg) 
        2. firing_rate_fig (firing_rate.jpg)
        3. dist_fig (distribution.jpg)
        3. feature_fig (feature.jpg)
        '''
        # visualization of Cl Chart:
        cl_fig=plt.figure()
        plt.title("Cl Chart")
        plt.xlabel("Adjust Times")
        plt.ylabel("Cl")
        for i in range(len(self.model)):
            cl_list=[x[i] for x in self.Cl_list]
            plt.plot(cl_list, label='synapse {}'.format(i+1))
        plt.legend()
        # Visualization of firing rates:
        firing_rate_fig=plt.figure()
        plt.title("Voltage Resetting Rate")
        plt.xlabel("Adjust Times")
        plt.ylabel("rate")
        n=len(self.history_rates[0])  # The amount of neuron layers
        for i in range(n):
            rates=[rate_list[i] for rate_list in self.history_rates]
            plt.plot(rates, label="Neurons {}".format(i+1))
        plt.legend()
        # Visualization of parameter distribution:
        dist_fig=plt.figure(figsize=(8, 6*len(self.model)))
        freq_list=self.dist_list[-1]
        x_label_list=['0.{}~0.{}'.format(i,i+1) for i in range(9)]
        x_label_list.append('0.9~1.0')
        for param_idx,single_list in enumerate(freq_list):
            plt.subplot(len(freq_list), 1, param_idx+1)
            plt.title("Distribution of Parameter {}".format(param_idx+1))
            plt.ylabel("Frequency")
            plt.bar(range(10), height=single_list)
            plt.xticks(ticks=range(10), labels=x_label_list, rotation=30)
        plt.tight_layout()
        # Visualization of Feature Map:
        feature_fig=plt.figure()
        if isinstance(self.model, CV_STDPModel):
            n_max=0
            for tens in self.model.log:  # Original shape of self.model.log: [sequence_number(4), batch_size, out_channels, rows, cols]
                n,x,y=tens[0].shape # n: amount of out channels. x: amount of rows. y: amout of columns
                if n_max<n:
                    n_max=n  # after the loop, n_max will reach to the biggest channel amount among all the layers
            for i,tens in enumerate(self.model.log):
                tens=tens[0] # Get the first output in a batch. Then Shape: [out_channels, rows, cols]
                for j,feature in enumerate(tens):  #feature shape: [x, y]
                    plt.subplot(n_max, 5, i+1+j*5)  # 4 pics in a row
                    feature=feature.cpu().numpy()
                    plt.imshow(feature, cmap='gray')
                    if i==0:
                        plt.title("DOG, feature "+str(j+1))
                    else:
                        plt.title("Conv "+str(i)+", feature "+str(j+1))
            plt.subplot(n_max, 5, 5)
            plt.title("Rate")
            print(self.ratelog[0].cpu().numpy())
            plt.imshow(self.ratelog[0].cpu().numpy(), cmap='gray')
            plt.tight_layout()
        if save==True:
            cl_fig.savefig(self.dir+"/cl.jpg")
            feature_fig.savefig(self.dir+"/feature.jpg")
            firing_rate_fig.savefig(self.dir+"/firing_rate.jpg")
            dist_fig.savefig(self.dir+"/distribution.jpg")
            print("STDPExe: visualization Charts saved.")
        return cl_fig, firing_rate_fig, dist_fig, feature_fig
    def test(self, save=True):
        print("STDPExe: start testing.")
        with torch.no_grad():
            test_fig=plt.figure(figsize=(10,12))
            data_iter=iter(self.test_data)
            for i in range(3):  # Select 3 data to do the test
                plt.subplot(3,1,i+1)
                plt.title("Sample {}".format(i+1))
                functional.reset_net(self.model)
                x,y=next(data_iter) # x shape: [batch_size, channels, time_step]
                x= x[0].cuda() if torch.cuda.is_available()==True else x[0]
                time_step=x.shape[-1]
                rates=[]
                for ts in range(time_step):
                    single_x=x[:,ts].view(1,-1,1)
                    out=self.model(single_x, True, False) # record_rates=True, auto_reset=False
                    rates.append(self.model.rates)
                n=len(rates[0])
                for j in range(n):
                    plt.plot(np.arange(time_step), [x[j] for x in rates], label="Neuron {}".format(j+1), linewidth=0.5)
                plt.legend()
                process_print(i+1,3)
            plt.tight_layout()
            if save==True:
                test_fig.savefig("{}/3_sample_test.jpg".format(self.dir))
                print("Chart saved")
            print("STDPExe: test finished.")
            return test_fig

