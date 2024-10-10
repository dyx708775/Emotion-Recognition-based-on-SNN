# By Xiangnan Zhang, School of Future Technologies, Beijing Institute of Technology
# Subpackage for Surrogate Gradient method
# Modified: 2024.7.24
# Dependencies: SpikingJelly, PyTorch

import spikingjelly.activation_based as sj
import torch
from typing import *
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from . import tools
from . import Regression
from . import StdpModel

class SCU_DNN_Combination(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scu = StdpModel.EEG_SequentialCompressionUnit()
        self.dnn = Regression.EEG_DNN()
    def forward(self, x):
        x = self.scu(x)
        print("after scu: {}".format(x.shape))
        pred = self.dnn(x)
        return pred

class SurrogateExe():
    def __init__(self, direc: str, model: torch.nn.Module, dataloaders: Tuple[torch.utils.data.DataLoader]):
        self.model=model
        self.dir=direc
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = torch.nn.BCELoss()
        self.train_ds, self.test_ds = dataloaders
        if torch.cuda.is_available():
            self.model=self.model.to("cuda")
            print("SurrogateExe: Use CUDA")
        else:
            print("SurrogateExe: Use CPU")
        self.loss_log={
            "train": [],
            "test": [] }
        self.accuracy_log={
            "train": [],
            "test": []}
        self.avg_grad=[]
    def load(self):
        loadmap='cpu'
        if torch.cuda.is_available()==True:
            loadmap='cuda'
        self.model.load_state_dict(torch.load(self.dir+'/Surrogate.pt', map_location=loadmap))
        print("SurrogateExe: Surrogate model loaded.")
    def test(self, test_length=256):
        batch_length = test_length/len(self.test_ds.dataset)
        batch_length = min(batch_length, len(self.test_ds))
        self.model.eval()
        with torch.no_grad():
            correct_num=0
            loss=0
            for i,(x,y) in enumerate(self.test_ds):
                if torch.cuda.is_available()==True:
                    x,y=x.cuda(),y.cuda() 
                out=self.model(x)
                loss+=self.loss_fn(out, y)
                out[out>=0.5]=1
                out[out<0.5]=0
                as_int = lambda x: x.int().item()
                for n,single_pred in enumerate(out):
                    if as_int(single_pred[0])==as_int(y[n][0]) and as_int(single_pred[1])==as_int(y[n][1]):
                        correct_num+=1.
                tools.process_print(i+1, batch_length)
                if i+1==batch_length:
                    break
            loss_mean=loss/batch_length
            loss_mean=loss_mean.cpu().item()
            accuracy=correct_num/len(self.test_data.dataset)
            self.loss_log['test'].append(loss_mean)
            self.accuracy_log['test'].append(accuracy)
            if do_print==True:
                print("Test: loss={}, accuracy={}".format(loss_mean, accuracy))
    def train(self, Epochs: int=1):
        '''
        length: the amount of data used to do the train. default: whole training dataset.
        do_print: whether print the result or not.
        '''
        print("SurrogateExe: start training.") 
        for epoch in range(Epochs):
            self.model.train()
            correct_num=0
            total_amount_for_accuracy=0
            for i,(x,y) in enumerate(self.train_ds):
                if torch.cuda.is_available()==True:
                    x,y=x.cuda(),y.cuda()
                pred=self.model(x)
                self.optimizer.zero_grad()
                loss=self.loss_fn(pred, y)
                self.loss_log['train'].append(loss.cpu().item())
                loss.backward()
                avg_grad_list=[]
                for param in self.model.parameters():
                    if param.grad is None:
                        avg_grad_list.append(None)
                    else:
                        grad=param.grad.flatten()
                        grad=torch.abs(grad)
                        avg_grad=grad.sum().cpu().item()/len(grad)
                        avg_grad_list.append(avg_grad)
                self.avg_grad.append(avg_grad_list)
                self.optimizer.step()
                pred[pred>=0.5]=1
                pred[pred<0.5]=0
                total_amount_for_accuracy+=len(y)
                as_int = lambda x: x.int().item()
                for n,single_pred in enumerate(pred):
                    if as_int(single_pred[0])==as_int(y[n][0]) and as_int(single_pred[1])==as_int(y[n][1]):
                        correct_num+=1.
                tools.process_print(i+1, len(self.train_ds))
                if (i%5==0 and i!=0) or i==len(self.train_ds)-1:
                    torch.save(self.model.state_dict(), self.dir+'/Surrogate.pt')
                    accu=correct_num/total_amount_for_accuracy
                    total_amount_for_accuracy=0
                    correct_num=0
                    self.accuracy_log['train'].append(accu)
                    print("\nEpoch {}, train loss = {}, train accuracy = {}".format(epoch+1, loss, accu))
                    print("Testing: ", end='')
                    self.test()
                    fig, grad_chart = self.visualize()
                    plt.close(fig)
                    plt.close(grad_chart)
                    with open(self.dir+'/Surrogate_process_record.txt', 'w') as file:
                        file.write("time={}\n".format(time.time()))
                        file.write("Epoch {}/{}, {}/{}\n".format(epoch+1, Epochs, i+1, len(self.train_ds)))
                        file.write("train_loss={}\n".format(self.loss_log['train']))
                        file.write("test_loss={}\n".format(self.loss_log['test']))
                        file.write("train_accuracy={}\n".format(self.accuracy_log['train']))
                        file.write("test_accuracy={}\n".format(self.accuracy_log['test']))
                        file.write("avg_grad={}\n".format(self.avg_grad))
        print("SurrogateExe: training finished.")        
    def visualize(self, save=True) -> matplotlib.figure.Figure:
        plt.rcParams['axes.unicode_minus']=False
        fig=plt.figure(figsize=(8,16))
        plt.subplot(4,1,1)
        plt.title('train loss')
        plt.plot(self.loss_log['train'])
        plt.xlabel("Adjustment times")
        plt.ylabel("Loss")
        plt.subplot(4,1,2)
        plt.title('train accuracy')
        plt.plot(self.accuracy_log['train'])
        plt.xlabel("Adjustment times")
        plt.ylabel("Accuracy")
        plt.subplot(4,1,3)
        plt.title('test loss')
        plt.plot(self.loss_log['test'])
        plt.xlabel("Testing times")
        plt.ylabel("Loss")
        plt.subplot(4,1,4)
        plt.title('test accuracy')
        plt.plot(self.accuracy_log['test'])
        plt.xlabel("Testing times")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        grad_chart=plt.figure()
        plt.title("Gradient Chart")
        plt.xlabel("Backward Time")
        tensor_num=len(self.avg_grad[0])
        for i in range(tensor_num):
            plt.plot([x[i] for x in self.avg_grad], label="tensor {}".format(i+1) if x[i] is not None else 0)
        plt.legend()
        plt.tight_layout()
        if save==True:
            fig.savefig(self.dir+'/surrogate_result.jpg')
            grad_chart.savefig(self.dir+'/grad.jpg')
            print("SurrogateExe: visualization chart saved.")
        return fig, grad_chart
