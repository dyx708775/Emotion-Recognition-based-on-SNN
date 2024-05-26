# Xiangnan Zhang 2024, School of Future Technologies, Beijing Institute of Technology
# modified: 2024.5.26

# Dependencies: PyTorch, MatPlotLib
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib

from SNN_tools import process_print
import SNN_StdpModel as stdp

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

class RegExe():
    def __init__(self, reg_model: RegressionModel, stdp_exe: stdp.STDPExe, store_dir: str, train_data, test_data, **argv):
        '''
        store_dir: directory of a folder that will store the regression model and features
        train_data and test_data: either packed as batches or not
        '''
        self.model=reg_model
        if torch.cuda.is_available()==True:
            self.model.to('cuda')
            print("Regression: use CUDA.")
        self.stdp_exe=stdp_exe
        self.stdp_exe.load()
        print("STDP model loaded")
        self.dir=store_dir
        self.train_data=train_data 
        self.test_data=test_data
        self.optimizer=optim.SGD(self.model.parameters(), lr=1e-3)
        self.loss_fn=nn.CrossEntropyLoss()
        self.loss_log={'train':[], 'test':[]}
        self.accuracy_log={'train':[], 'test':[]}
    def __forward(self, x):
        '''
        This method is aimed to connect STDP model and the regression model.
        x: images without normalization, torch.Tensor, shape [n, 28, 28]
        return: logits, torch.Tensor, shape [n, 10]
        '''
        if torch.cuda.is_available()==True:
            x=x.cuda()
        with torch.no_grad():
            inp=self.stdp_exe.forward(x) # Send to STDP model, time steps is defined previously
            inp.unsqueeze(1) # shape of inp: [n, 1, 4, 4]
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
        self.model.eval()
        with torch.no_grad():
            correct_num=0
            loss=0
            for i,(x,y) in enumerate(self.test_data):
                if torch.cuda.is_available()==True:
                    x,y=x.cuda(),torch.tensor([y]).cuda() # shape of y: (,n)
                pred=self.__forward(x)
                loss+=self.loss_fn(pred, y)
                correct_num+=(pred.argmax(1)==y).float().sum()
                process_print(i+1, length if length!=None else len(self,test_data))
                if i+1==length:
                    break
            loss_mean=loss/len(self.test_data)
            loss_mean=loss_mean.cpu().item()
            accuracy=correct_num/len(self.test_data)
            accuracy=accuracy.cpu.item()
            self.loss_log['test'].append(loss_mean)
            self.accuracy_log['test'].append(accuracy)
            if do_print==True:
                print("Test: loss={}, accuracy=[]".format(loss_mean, accuracy))
    def train(self, Epochs: int=1, length=None, do_print==True):
        '''
        length: the amount of data used to do the train. default: whole training dataset.
        do_print: whether print the result or not.
        ''' 
        for epoch in range(Epochs):
            self.model.train()
            correct_num=0
            recorder # record the last time that culculate the accuracy
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
                process_print(i+1, length if length!=None else len(self.train_data))
                if (i%1000==0 and i!=0) or i==len(self.train_data)-1:
                    accu=correct_num/(i-recorder+1) # calculate the accuracy for every 1000 times or less
                    recorder=i
                    self.accuracy_log['train'].append(accu.cpu().item())
                    if do_print==True:
                        print("Epoch {}, train loss = {}, train accuracy = {}".format(epoch+1, loss, accu))
                if i+1==length:
                    break
            torch.save(self.model.state_dict(), self.dir+'/regression.pt')
    def load(self, load_stdp=True):
        '''
        Used to load the state dict of the regression model, as well as STDP model.
        load_stdp: whether the state dict of STDP model should be loaded at the same time.
        '''
        loadmap='cpu'
        if torch.cuda.is_available()==True:
            loadmap='cuda'
        self.model.load_state_dict(torch.load(self.dir+'/regression.pt', map_location=loadmap))
        if load_stdp==True:
            self.stdp_exe.load()
    def visualize(self, save=True) -> matplotlib.figure.Figure:
        plt.rcParams['axes.unicode_minus']=False
        fig=plt.figure()
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
