# Xiangnan Zhang 2024, School of Future Technologies, Beijing Institute of Technology
# modified: 2024.6.1
# Dependencies: PyTorch, MatPlotLib

# This is the definition of a regression classifier used after the unsupervised SNN Network

import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib
import time

from .tools import process_print

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Conv2d(1,10,4)
    def forward(self, x):
        '''
        x: shape as [n, 4. 4]
        return: logits with shape as [n, 10]
        '''
        x=self.conv(x)
        x=x.view(-1,10)
        return x

class EEGReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1000,2)
        self.sigmoid=nn.Sigmoid()
    def forward(self, state):
        '''
        in: [batch_size, 1000]
        out: [batch_size, 2]
        '''
        x=self.linear(state)
        pred=self.sigmoid(x)
        return pred

class RegExe():
    def __init__(self, reg_model: RegressionModel, unsup_exe, store_dir: str, train_data, test_data, **argv):
        '''
        unsup_exe: unsupervised SNN Exe class. Notice that .forward() method should be defined
        store_dir: directory of a folder that will store the regression model and features
        train_data and test_data: either packed as batches or not
        '''
        self.model=reg_model
        if torch.cuda.is_available()==True:
            self.model.to('cuda')
            print("Regression: use CUDA.")
        self.unsup_exe=unsup_exe
        try:
            self.unsup_exe.load()
            print("STDP model loaded")
        except FileNotFoundError:
            print("Warning: STDP model doesn't exist.")
        self.dir=store_dir
        self.train_data=train_data 
        self.test_data=test_data
        self.optimizer=optim.SGD(self.model.parameters(), lr=1e-3)
        self.loss_fn=nn.MSELoss()
        self.loss_log={'train':[], 'test':[]}
        self.accuracy_log={'train':[], 'test':[]}
    def __forward(self, x):
        '''
        This method is aimed to connect STDP model and the regression model.
        '''
        if torch.cuda.is_available()==True:
            x=x.cuda()
        inp=self.unsup_exe.forward(x) # Send to STDP model, time steps is defined previously
        pred=self.model(inp) # Send to Regression model
        return pred
    def __call__(self, x, logits=False):
        pred=self.__forward(x)
        if logits==False:
            return pred.argmax(1)
        else:
            return pred
    def test(self, length=None, do_print=True):
        '''
        Get the accuracy and average loss on testing dataset.
        length: the amount of data used to do the test. default: whole testing dataset.
        do_print: whether print the result or not.
        Once it has been processed, the result will be stored in self.loss_log and self.accuracy_log.
        '''
        if length==None:
            length=len(self.test_data)
        self.model.eval()
        with torch.no_grad():
            correct_num=0
            loss=0
            for i,(x,y) in enumerate(self.test_data):
                if torch.cuda.is_available()==True:
                    x,y=x.cuda(),torch.tensor([y]).cuda() # shape of y: (,n)
                pred=self.__forward(x)
                loss+=self.loss_fn(pred, y)
                pred[pred>=0.5]=1
                pred[pred<0.5]=0
                for n,single_pred in enumerate(pred):
                    if single_pred[0]==y[n][0] and single_pred[1]==y[n][1]:
                        correct_num+=1
                process_print(i+1, length)
                if i+1==length:
                    break
            loss_mean=loss/length
            loss_mean=loss_mean.cpu().item()
            accuracy=float(correct_num)/length
            self.loss_log['test'].append(loss_mean)
            self.accuracy_log['test'].append(accuracy)
            if do_print==True:
                print("Test: loss={}, accuracy={}".format(loss_mean, accuracy))
    def train(self, Epochs: int=1, length=None, do_print=True):
        '''
        length: the amount of data used to do the train. default: whole training dataset.
        do_print: whether print the result or not.
        ''' 
        if length==None:
            length=len(self.train_data)
        for epoch in range(Epochs):
            self.model.train()
            correct_num=0
            recorder=0 # record the last time that culculate the accuracy
            for i,(x,y) in enumerate(self.train_data):
                if torch.cuda.is_available()==True:
                    x,y=x.cuda(),torch.tensor([y]).cuda()
                pred=self.__forward(x)
                self.optimizer.zero_grad()
                loss=self.loss_fn(pred, y)
                self.loss_log['train'].append(loss.cpu().item())
                loss.backward()
                self.optimizer.step()
                correct_num+=(pred.argmax(1)==y).float().sum()
                process_print(i+1, length)
                if (i%100==0 and i!=0) or i==length-1:
                    torch.save(self.model.state_dict(), self.dir+'/regression.pt')
                    accu=correct_num/(i-recorder+1) # calculate the accuracy for every 1000 times or less
                    recorder=i
                    self.accuracy_log['train'].append(accu.cpu().item())
                    if do_print==True:
                        print("\nEpoch {}, train loss = {}, train accuracy = {}".format(epoch+1, loss, accu))
                        print("Testing: ", end='')
                        self.test(do_print=True)
                    else:
                        self.test(do_print=False)
                    self.visualize()
                    correct_num=0
                    with open(self.dir+'/Reg_process_record.txt', 'w') as file:
                        file.write("time={}\n".format(time.time()))
                        file.write("Epoch {}/{}, {}/{}\n".format(epoch+1, Epochs, i+1, length))
                        file.write("train_loss={}\n".format(self.loss_log['train']))
                        file.write("test_loss={}\n".format(self.loss_log['test']))
                        file.write("train_accuracy={}\n".format(self.accuracy_log['train']))
                        file.write("test_accuracy={}\n".format(self.accuracy_log['test']))
                if i+1==length:
                    break
    def load(self, load_unsup=True):
        '''
        Used to load the state dict of the regression model, as well as STDP model.
        load_unsup: whether the state dict of unsupervised SNN model should be loaded at the same time.
        '''
        loadmap='cpu'
        if torch.cuda.is_available()==True:
            loadmap='cuda'
        self.model.load_state_dict(torch.load(self.dir+'/regression.pt', map_location=loadmap))
        if load_unsup==True:
            self.unsup_exe.load()
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
        if save==True:
            fig.savefig(self.dir+'/reg_result.jpg')
        return fig
