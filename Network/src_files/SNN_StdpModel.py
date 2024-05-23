# By Xiangnan Zhang, School of Future Technologies, Beijing Institute of Technology
# This is the definition of the STDP-based model in Affective Computing Program
# Dependence: SpikingJelly, PyTorch, NumPy, SciPy, MatPlotLib
# Modified: 2024.5.18

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

from SNN_tools import process_print, Logger

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
        self.log=[None, None, None, None]
        self.record=False  # Whether record the propogation into self.log. This will work in forward() #
        for param in self.parameters():
            param.data=torch.abs(param.data)
    def __getitem__(self, index):
        lis=[self.node1, self.conv1, self.node2, self.conv2, self.node3, self.conv3, self.node4]
        return lis[index]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
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

def w_dep_factor(a: float, w: torch.Tensor) ->torch.Tensor:
    '''
    Weight Dependence Factor for STDPLearner. More details
    can be found in Kheradpisheh et al.(2018)
    It should be preprocessed by lambda.
    '''
    return a*w*(1-w)
    
class STDPExe():
    def __init__(self, model: STDPModel, store_dir: str, train_data, test_data, **argv):
        '''
        store_dir refers to a directory f a folder, which includes file(s):
        1. STDPModel.pt  Weights of model.
        train_data and test_data: iterable, return an (x, y) tuple
        '''
        self.dir=store_dir
        if torch.cuda.is_available()==True:
            print("Use CUDA")
            self.device=torch.device("cuda")
        else:
            print("Use CPU")
            self.device=torch.device("cpu")
        self.model=model.to(self.device)
        self.train_data=train_data
        self.test_data=test_data
        self.optim=torch.optim.SGD(self.model.parameters(), lr=1e-1, momentum=0)
        self.pre_wdf=lambda w: w_dep_factor(3e-3,w)
        self.post_wdf=lambda w: w_dep_factor(4e-3,w)
        self.stdp_learners=[
            learning.STDPLearner('s', self.model[i], self.model[i+1], 2, 2, self.pre_wdf, self.post_wdf) for i in [1,3,5]
            ] #Three STDPLearner for total
        self.Cl_list=[[],[],[]]
        self.ratelog=None
        if 'T' in argv:
            self.T=argv['T']
        else:
            self.T=30
        self.logger=Logger(self)
    def forward(self, x: torch.Tensor, record=False) -> torch.Tensor:
        '''
        x: a batch of tensors that will be processed into self.model, [n, 28, 28]
        T: periods for a single propogation
        return: firing rate for each pixel, [n, 2]
        '''
        T=self.T
        x=x.view(-1,1,28,28)
        with torch.no_grad():
            if self.device==torch.device("cuda"):
                x=x.cuda()
            functional.reset_net(self.model)
            for learner in self.stdp_learners:
                learner.reset()
            pred=0
            for i in range(T):
                if i==T-1 and record==True:
                    self.model.record=True
                else:
                    self.model.record=False
                pred+=self.model(x)
                for learner in self.stdp_learners:
                    learner.step(on_grad=True)
            return pred/T
    def load(self):
        loadmap='cpu'
        if torch.cuda.is_available()==True:
            loadmap='cuda'
        self.model.load_state_dict(torch.load(self.dir+"/STDPModel.pt", map_location=loadmap))
    def train(self, epochs: int, save=True):
        for epoch in range(epochs):
            self.model.train()
            for index,(x,y) in enumerate(self.train_data):
                self.optim.zero_grad()
                fire_rate=self.forward(x)
                self.optim.step()
                with torch.no_grad():
                    for Cl_i,i in enumerate([1,3,5]):
                        length=0
                        par_sum=0
                        for param in self.model[i].parameters():
                            param=param.data.flatten()
                            sub=(param*(1-param)).sum()
                            par_sum+=sub
                            length+=len(param)
                        cl=par_sum/length
                        cl=cl.item()
                        self.Cl_list[Cl_i].append(cl)
                process_print(index+1, len(self.train_data))
                if self.Cl_list[0][-1]<=1e-5 and self.Cl_list[1][-1]<=1e-5 and self.Cl_list[2][-1]<=1e-5:
                    print("The Model has been converged.")
                    break
            self.logger.write_log()
            if save==True:
                torch.save(self.model.state_dict(), self.dir+"/STDPModel.pt")
            self.model.eval()
            x,y=self.test_data[0]
            pred=self.forward(x, True)
            self.ratelog=pred
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
