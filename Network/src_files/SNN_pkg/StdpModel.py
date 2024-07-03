# By Xiangnan Zhang, School of Future Technologies, Beijing Institute of Technology
# This is the definition of the STDP-based model in Affective Computing Program
# Dependency: SpikingJelly, PyTorch, NumPy, SciPy, MatPlotLib
# Modified: 2024.6.14

from spikingjelly.activation_based import neuron,layer,functional,learning
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor,ToPILImage
import numpy as np
from scipy import ndimage
from typing import *
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

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

class EEG_STDPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1_linear=layer.Linear(14,20,bias=False)
        self.b1_lif=neuron.LIFNode(v_threshold=5.)
        self.b1_cache=0
        self.b1_recurrent=layer.Linear(20,20,bias=False)
        self.b2_linear=layer.Linear(20,20,bias=False)
        self.b2_lif=neuron.LIFNode(v_threshold=5.)
        self.b2_cache=0
        self.b2_recurrent=layer.Linear(20,20,bias=False)
        self.b3_linear=layer.Linear(20,500,bias=False)
        self.b3_lif=neuron.LIFNode(v_threshold=9999999999999999.)
        self.b3_cache=0
        self.b3_recurrent=layer.Linear(500,500,bias=False)
        self.rates=[]  # Used to record the firing rates of each LIF layer while training.
        self.synapse_list=[
            [self.b1_linear, self.b1_lif],
          #  [self.b1_recurrent, self.b1_lif],
            [self.b2_linear, self.b2_lif],
          #  [self.b2_recurrent, self.b2_lif],
            [self.b3_linear, self.b3_lif],
          #  [self.b3_recurrent, self.b3_lif]
          ]
        for param in self.parameters():
            torch.nn.init.normal_(param.data, mean=0.8, std=0.5)
            param.data=torch.clamp(param.data, min=0, max=1)
   #    param=next(self.b3_recurrent.parameters())
   #    torch.nn.init.normal_(param.data, mean=0.2, std=0.5)
  #     param.data=torch.clamp(param.data, min=0, max=1)
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
        self.b1_cache=torch.zeros(batch_size,20)
        self.b2_cache=torch.zeros(batch_size,20)
        self.b3_cache=torch.zeros(batch_size,500)
        if torch.cuda.is_available()==True:
            self.b1_cache=self.b1_cache.cuda()
            self.b2_cache=self.b2_cache.cuda()
            self.b3_cache=self.b3_cache.cuda()
        if auto_reset==True:
            functional.reset_net(self)
        for i in range(time_step):
            with torch.no_grad():
                inp=x[:,:,i]  # shape: [batch_size, 14]
            inp=self.b1_linear(inp)
         #  y_cache=self.b1_recurrent(self.b1_cache)
            inp=self.b1_lif(inp)
            self.b1_cache=inp
            inp=self.b2_linear(inp)
         #  y_cache=self.b2_recurrent(self.b2_cache)
            inp=self.b2_lif(inp)
            self.b2_cache=inp
            inp=self.b3_linear(inp)
          # y_cache=self.b3_recurrent(self.b3_cache)
            inp=self.b3_lif(inp)
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
        liquid_state=torch.exp(self.b3_lif.v)
        return liquid_state

def w_dep_factor(a: float, w: torch.Tensor) ->torch.Tensor:
    '''
    Weight Dependence Factor for STDPLearner. More details
    can be found in Kheradpisheh et al.(2018)
    It should be preprocessed by lambda.
    '''
    return a*w*(1-w)
    
class STDPExe():
    '''
    The following attributes are expected to be accessed in public:
    1. dir: if you want to change the saving directory.
    2. train_data
    3. self_data
    4.Cl_list: storing all the Cl values during training process.
    The following methods are expected to be called in public:
    1. forward(x)
    2. load(): load the state directory of the STDP model.
    3. calc_Cl(): to calculate the current Cl values.
    4. train()
    5. visualize()
    '''
    def __init__(self, model: nn.Module, store_dir: str, train_data: Iterable[Tuple], test_data: Iterable[Tuple]):
        '''
        store_dir refers to a directory of a folder, where to save training results and the model.
        train_data and test_data: iterable, return an (x, y) tuple. They should be packed as batches.
        '''
        self.dir=store_dir
        if torch.cuda.is_available()==True:
            print("STDP: Use CUDA")
            self.device=torch.device("cuda")
        else:
            print("STDP: Use CPU")
            self.device=torch.device("cpu")
        self.model=model.to(self.device)
        self.train_data=train_data
        self.test_data=test_data
        self.optim=torch.optim.SGD(self.model.parameters(), lr=1e-1, momentum=0)
        self.pre_wdf=lambda w: w_dep_factor(3e-3,w)  # 'wdf' means weight-dependence factor, see Morrison et al. (2018)
        self.post_wdf=lambda w: w_dep_factor(4e-3,w)
        self.stdp_learners=[
            learning.STDPLearner('s', self.model[i][0], self.model[i][1], 2, 2, self.pre_wdf, self.post_wdf) for i in range(len(self.model))
            ] 
        self.Cl_list=[] # Used to store all the Cl value during training
        self.logger=Logger(self)
        self.history_rates=[]  # histroy firing rates of each neuron layers during training.
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
        Load STDP model's state directory from the directory.
        '''
        location='cpu'
        if torch.cuda.is_available()==True:
            location='cuda'
        self.model.load_state_dict(torch.load(self.dir+"/STDPModel.pt", map_location=location))
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
    def train(self, epochs: int, save=True):
        '''
        save: Bool, wether save model's state dictionary
        After launching, you will get following files under store_dir:
        1. logs (txt files):
            1.1 STDP_log.txt: a log file about basic trainning setting. Updated for each epoch.
            1.2 STDP_process_record.txt: a log file recording training process. Updated during each epoch.
        2. charts (jpg files):
            2.1 cl.jpg: a chart recording the alternation of Cls.
            2.2 firing_rate.jpg: a chart recording the firing rates of each neuron layer. 
            2.3 feature.jpg: a graphic recording forward process if it is defined in visualize().
        3. STDPModel.pt: the state directory of the STDP model.
        '''
        for epoch in range(epochs):
            self.model.train()
            for index,(x,y) in enumerate(self.train_data):
                self.optim.zero_grad()
                fire_rate=self.forward(x, True) # set record_rate=True, record the firing rates.
                self.optim.step()
                with torch.no_grad():
                    self.Cl_list.append(self.calc_Cl())
                process_print(index+1, len(self.train_data))
                B_list=[cl<=1e-5 for cl in self.Cl_list[-1]]
                if (index%5==0 and index!=0) or index==len(self.train_data)-1 or all(B_list):
                    print("")
                    with open(self.dir+'/STDP_process_record.txt', 'w') as file:
                        file.write("time={}\n".format(time.time()))
                        file.write("epoch {}/{}, {}/{}\n".format(epoch+1, epochs, index+1, len(self.train_data)))
                        file.write("Cl={}\n".format(self.Cl_list))
                        file.write("spiking_rates={}\n".format(self.history_rates))
                    figs=self.visualize(True)
                    plt.close()
                    if save==True:
                        torch.save(self.model.state_dict(), self.dir+"/STDPModel.pt")
                        print("Model saved.")
                if all(B_list):  # Whether all the Cl values are less than 1e-5
                    print("The Model has converged.")
                    break
            self.logger.write_log()
    def visualize(self, save=True) -> tuple:
        plt.rcParams['axes.unicode_minus']=False
        '''
        The following charts will be created:
        1. cl_fig (cl.jpg) 
        2. firing_rate_fig (firing_rate.jpg)
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
        plt.title("Firing Rates")
        plt.xlabel("Adjust Times")
        plt.ylabel("rate (%)")
        n=len(self.history_rates[0])  # The amount of neuron layers
        for i in range(n):
            rates=[rate_list[i] for rate_list in self.history_rates]
            plt.plot(rates, label="Neurons {}".format(i+1))
        plt.legend()
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
            print("STDP Charts saved.")
        return cl_fig, firing_rate_fig, feature_fig
    def test(self, save=True):
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
                    plt.plot(np.arange(time_step-7800)+7800, [x[j] for x in rates[7800:]], label="Neuron {}".format(j+1), linewidth=0.5)
                plt.legend()
                process_print(i+1,3)
            plt.tight_layout()
            if save==True:
                test_fig.savefig("{}/3_sample_test.jpg".format(self.dir))
                print("Chart saved")
            return test_fig

