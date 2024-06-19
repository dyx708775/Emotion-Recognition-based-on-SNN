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

class STDPModel(nn.Module):
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
    def __single_forward(self, x: torch.Tensor) -> torch.Tensor:
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
        for i,sequence in enumerate(block):
            for lay in sequence:
                x=lay(x)
            if self.record==True:
                self.log[i+1]=x
        return x.view(-1,4,4)
    def get_Cl(self):
        Cl_list=[]
        for i in range(len(self)):
            layer_params=self[i][0].parameters()
            cl=0
            length=0
            for param in layer_params:
                param=param.data.flatten()
                length+=len(param)
                cl+=torch.sum(param*(1-param))
            cl=cl/length
            Cl_list.append(cl)
        return Cl_list
    def forward(self, x: torch.Tensor) -> torch.Tensor:
       with torch.no_grad():
           x=x.view(-1,1,28,28)
       functional.reset_net(self)
       pred=0
       T=30
       for i in range(T):
           pred+=self.__single_forward(x)
       return (pred/T).view(-1, 4, 4)

def w_dep_factor(a: float, w: torch.Tensor) ->torch.Tensor:
    '''
    Weight Dependence Factor for STDPLearner. More details
    can be found in Kheradpisheh et al.(2018)
    It should be preprocessed by lambda.
    '''
    return a*w*(1-w)
    
class STDPExe():
    def __init__(self, model: nn.Module, store_dir: str, train_data: Iterable[Tuple], test_data: Iterable[Tuple], **argv):
        '''
        store_dir refers to a directory f a folder, which includes file(s):
        1. STDPModel.pt  Weights of model.
        train_data and test_data: iterable, return an (x, y) tuple
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
        self.optim=torch.optim.SGD(self.model.parameters(), lr=1, momentum=0)
        self.pre_wdf=lambda w: w_dep_factor(3e-3,w)  # 'wdf' means weight-dependence factor, see Morrison et al. (2018)
        self.post_wdf=lambda w: w_dep_factor(4e-3,w)
        self.stdp_learners=[
            learning.STDPLearner('s', self.model[i][0], self.model[i][1], 2, 2, self.pre_wdf, self.post_wdf) for i in range(len(self.model))
            ] 
        self.Cl_list=[] # Used to store all the Cl value during training
        self.logger=Logger(self)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x: a batch of tensors that will be processed into self.model, [n, 28, 28]
        return: firing rate for each pixel, [n, 4, 4]
        '''
        if torch.cuda.is_available()==True:
            x=x.cuda()
        for learner in self.stdp_learners:
            learner.reset()  # With the reset-step pair, STDPLearner will not store extra delta_w, even in testing process
        pred=self.model(x)
        for learner in self.stdp_learners:
            learner.step(on_grad=True)
        return pred
    def load(self):
        location='cpu'
        if torch.cuda.is_available()==True:
            location='cuda'
        self.model.load_state_dict(torch.load(self.dir+"/STDPModel.pt", map_location=location))
    def train(self, epochs: int, save=True):
        '''
        save: Bool, wether save model's state dictionary
        '''
        for epoch in range(epochs):
            self.model.train()
            for index,(x,y) in enumerate(self.train_data):
                self.optim.zero_grad()
                fire_rate=self.forward(x)
                self.optim.step()
                with torch.no_grad():
                    self.Cl_list.append(self.model.get_Cl())
                process_print(index+1, len(self.train_data))
                B_list=[cl<=1e-5 for cl in self.Cl_list[-1]]
                if all(B_list):  # Whether all the Cl values are less than 1e-5
                    print("The Model has converged.")
                    break
            self.logger.write_log()
            if save==True:
                torch.save(self.model.state_dict(), self.dir+"/STDPModel.pt")
    def visualize(self, save=True) -> tuple:
        cl_fig=plt.figure()
        # visualization of Cl Chart:
        for i,cl_list in enumerate(self.Cl_list):
            plt.plot(cl_list, label="Conv "+str(i+1))
        plt.title("Cl Chart")
        plt.xlabel("Adjust Times")
        plt.ylabel("Cl")
        plt.legend()
        # Visualization of Feature Map:
        feature_fig=plt.figure()
        n_max=0
        for tens in self.model.log:  # Original shape of self.model.log: [batch_size, feature_num, x, y]
            n,x,y=tens[0].shape
            if n_max<n:
                n_max=n
        for i,tens in enumerate(self.model.log):
            tens=tens[0] # Then Shape: [n, x, y]
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
        return cl_fig, feature_fig
    def test(self, dataset='test', save=True):
        inp=[[],[],[],[],[],[],[],[],[],[]]
        data=self.test_data if dataset=='test' else self.train_data
        for label in range(10):
            counter=0
            i=0
            while True:
               x,y=data[i]
               if y==label:
                   inp[label].append(x)
                   counter+=1
               if counter==3:
                   break
               i+=1
        print("Data Loaded")
        fig=plt.figure(figsize=(12,6))
        for i,input_data in enumerate(inp):
            for j,img in enumerate(input_data):
                plt.subplot(3,10,i+1+10*j)
                if j==0:
                    plt.title(str(i))
                rate=self.forward(img)
                rate=rate.squeeze().numpy()
                plt.imshow(rate, cmap='gray')
        plt.tight_layout
        if save==True:
            fig.savefig(self.dir+'/test.jpg')
        return fig
