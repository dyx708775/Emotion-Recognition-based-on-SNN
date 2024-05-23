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
        x=self.conv(x)
        x=x.view(-1,10)
        return x

class RegExe():
    def __init__(self, reg_model: RegressionModel, stdp_exe: stdp.STDPExe, store_dir: str, train_data, test_data, **argv):
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
        x: torch.Tensor, shape [n, 28, 28]
        return: torch.Tensor, shape [n, 10]
        '''
        if torch.cuda.is_available()==True:
            x=x.cuda()
        with torch.no_grad():
            inp=self.stdp_exe.forward(x) # Send to STDP model
            inp.unsqueeze(1)
        pred=self.model(inp) # Send to Regression model
        return pred
    def __call__(self, x):
        pred=self.__forward(x)
        return pred.argmax(1)
    def test(self):
        self.model.eval()
        with torch.no_grad():
            correct_num=0
            loss=0
            for x,y in self.test_data:
                if torch.cuda.is_available()==True:
                    x,y=x.cuda(),torch.tensor([y]).cuda()
                pred=self.__forward(x)
                loss+=self.loss_fn(pred, y)
                self.loss_log['test'].append(loss)
                correct_num+=(pred.argmax(1)==y).float().sum()
            loss_mean=loss/len(self.test_data)
            accuracy=correct_num/len(self.test_data)
            self.loss_log['test'].append(loss_mean.cpu().item())
            self.accuracy_log['test'].append(accuracy.cpu.item())
            print("Test: loss={}, accuracy=[]".format(loss_mean, accuracy))
    def train(self, Epochs: int):
        for epoch in range(Epochs):
            self.model.train()
            correct_num=0
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
                process_print(i+1, len(self.train_data))
                if (i%1000==0 and i!=0) or i==len(self.train_data)-1:
                    accu=correct_num/(i+1)
                    self.accuracy_log['train'].append(accu.cpu().item())
                    print("Epoch {}, train loss = {}, train accuracy = {}".format(epoch+1, loss, accu))
            torch.save(self.model.state_dict(), self.dir+'/regression.pt')
    def load(self, load_stdp=True):
        loadmap='cpu'
        if torch.cuda.is_available()==True:
            loadmap='cuda'
        self.model.load_state_dict(torch.load(self.dir+'/regression.pt', map_location=loadmap))
        if load_stdp==True:
            self.stdp_exe.load()
    def visualize(self, save=True) -> matplotlib.figure.Figure:
        fig=plt.figure()
        plt.subplot(4,1,1)
        plt.title('train loss')
        plt.plot(self.loss_log['train'])
        plt.subplot(4,1,2)
        plt.title('train accuracy')
        plt.plot(self.accuracy_log['train'])
        plt.subplot(4,1,3)
        plt.title('test loss')
        plt.plot(self.loss_log['test'])
        plt.subplot(4,1,4)
        plt.title('test accuracy')
        plt.plot(self.accuracy_log['test'])
        plt.tight_layout()
        if save==True:
            fig.savefig(self.dir+'/reg_result.jpg')
        return fig
