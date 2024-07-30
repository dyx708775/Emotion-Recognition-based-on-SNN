# Xiangnan Zhang 2024, School of Future Technologies, Beijing Institute of Technology
# modified: 2024.6.1
# Dependencies: PyTorch, MatPlotLib

# This is the definition of a regression classifier used after the unsupervised SNN Network

import torch
from torch import nn, optim
import matplotlib.pyplot as plt
import matplotlib
import time
import math
import ast
from typing import *

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



class EEG_Reg(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(32,2)
        self.sigmoid=nn.Sigmoid()
    def forward(self, state):
        '''
        in: [batch_size, 32]
        out: [batch_size, 2]
        '''
        state=(2*state-(math.exp(3)+1))/(math.exp(3)-1)        
        state=self.linear(state)
        pred=self.sigmoid(state)
        return pred



class EEG_DNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1=nn.Linear(128,1000)
        self.batch_norm_1=nn.BatchNorm1d(1000)
        self.ReLU_1=nn.ReLU()
        self.linear_2=nn.Linear(1000,2)
        self.batch_norm_2=nn.BatchNorm1d(2)
        self.sigmoid=nn.Sigmoid()
    def forward(self, state):
        '''
        in: [batch_size, 32]
        out: [batch_size, 2]
        '''
        state=(2*state-(math.exp(3)+1))/(math.exp(3)-1)
        b1_state=self.linear_1(state)
        b1_state_after_linear=b1_state
        b1_state=self.batch_norm_1(b1_state)
        b1_state=self.ReLU_1(b1_state)
        b2_state=self.linear_2(b1_state)+b1_state_after_linear.abs().mean()
        b2_state=self.batch_norm_2(b2_state)
        pred=self.sigmoid(b2_state)
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
            print("RegExe: use CUDA.")
        else:
            self.model.to('cpu')
            print("RegExe: use CPU.")
        self.unsup_exe=unsup_exe
        try:
            self.unsup_exe.load()
        except FileNotFoundError:
            print("RegExe Warning: STDP model doesn't exist.")
        self.dir=store_dir
        self.train_data=train_data 
        self.test_data=test_data
        self.optimizer=optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss_fn=nn.BCELoss()
        self.loss_log={'train':[], 'test':[]}
        self.accuracy_log={'train':[], 'test':[]}
        self.avg_grad=[]


    def with_record(self):
        '''
        Used to load training record from Reg_process_record.
        Training record includes:
            loss(train, test), accuracy(train, test)
        '''
        with open("{}/Reg_process_record.txt".format(self.dir), 'r') as file:
            content=file.read()
            content=content.split('\n')
        get_list = lambda i: ast.literal_eval((content[i].split('='))[1])

        self.loss_log["train"]=get_list(2)
        self.loss_log["test"]=get_list(3)
        self.accuracy_log["train"]=get_list(4)
        self.accuracy_log["test"]=get_list(5)

        print("RegExe: current record length:")
        print("    train: {}, {}".format(len(self.loss_log["train"]), len(self.accuracy_log["train"])))
        print("    test: {}, {}".format(len(self.loss_log["test"]), len(self.accuracy_log["test"])))


    def __forward(self, x):
        '''
        This method is aimed to connect Unsupervised model and the classifier during forward propogating process.
        Users should use __call__ method instead.
        '''
        if torch.cuda.is_available()==True:
            x=x.cuda()
        inp=self.unsup_exe.forward(x) # Send to STDP model, time steps is defined previously
        pred=self.model(inp) # Send to Regression model
        return pred


    def __call__(self, x, logits=False):
        '''
        Forward propogating with both unsupervised model and the classifier.
        '''
        pred=self.__forward(x)
        if logits==False:
            pred[pred>=0.5]=1.
            pred[pred<0.5]=0.
            return pred
        else:
            return pred


    def test(self, testing_length: int = None) -> Tuple[int, float, float] :
        '''
        Get the accuracy and average loss on testing dataset.
        testing_length: the amount of data (not the length of dataloader) used to do the test. default None (the minimum between 128 and the length of testing dataset, but not the length of dataloader).
        return: actual_testing_length, loss_mean, accuracy.
        NOTICE:
            1. While using this method, you don't need to worry about balancing testing_length and the length of your Dataset object. It will always automatically select the less one.
            2. Every time using this method, the order of testing dataset will be shuffled if using tools.SimpleDeap to load your data, meaning that if testing_length is less than the length of your Dataset object (length = len(self.test_data.dataset)), data used to do the test will not be the same. 
        '''
        self.test_data.dataset.should_shuffle=True

        if testing_length==None:
            length=min(int(128/self.test_data.batch_size), len(self.test_data))
        else:
            length = min(int(testing_length/self.test_data.batch_size), len(self.test_data))

        self.model.eval()
        with torch.no_grad():
            correct_num=0
            loss=0
            for i,(x,y) in enumerate(self.test_data):
                if torch.cuda.is_available()==True:
                    x,y=x.cuda(),y.cuda() # shape of y: (,n)
                pred=self.__forward(x)
                loss+=self.loss_fn(pred, y)
                pred[pred>=0.5]=1
                pred[pred<0.5]=0
                as_int = lambda x: x.int().item()
                for n,single_pred in enumerate(pred):
                    if as_int(single_pred[0])==as_int(y[n][0]) and as_int(single_pred[1])==as_int(y[n][1]):
                        correct_num+=1.
                process_print(i+1, length)
                if i+1==length:
                    break
            loss_mean=loss/length
            loss_mean=loss_mean.cpu().item()
            accuracy=correct_num/(length*self.test_data.batch_size)

        return length*self.test_data.batch_size, loss_mean, accuracy


    def train(self, Epochs: int=1, **argv) -> None:
        '''
        Parameters in argv:
        1. early_stop_accuracy: Tuple[float,float]. This method need two threshold accuracy for two level early stop test. Default (0.48, 0.45).
        2. test_length: int, default None
        '''
        print("RegExe: start training.") 

        length=len(self.train_data)
        early_stop = False
        early_stop_accuracy_tuple = argv["early_stop_accuracy"] if "early_stop_accuracy" in argv else (0.48, 0.45)
        early_stop_threshold_v1, early_stop_threshold_v2 = early_stop_accuracy_tuple
        if early_stop_threshold_v1 >= 1 or early_stop_threshold_v2 >=1:
            raise ValueError("Early stop threshold accuracy should be less than 1. Now {} and {}.".format(*early_stop_accuracy_tuple))
        test_length = argv["test_length"] if "test_length" in argv else None

        for epoch in range(Epochs):
            self.model.train()

            correct_num=0
            total_amount_for_accuracy=0

            for i,(x,y) in enumerate(self.train_data):

                if torch.cuda.is_available()==True:
                    x,y=x.cuda(),y.cuda()

                pred=self.__forward(x)
                self.optimizer.zero_grad()
                loss=self.loss_fn(pred, y)
                self.loss_log['train'].append(loss.cpu().item())
                loss.backward()

                avg_grad_list=[]
                for param in self.model.parameters():
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

                process_print(i+1, length)

                if (i%5==0 and i!=0) or i==length-1:
                    # Statistics

                    torch.save(self.model.state_dict(), self.dir+'/regression.pt')

                    accu=correct_num/total_amount_for_accuracy
                    total_amount_for_accuracy=0
                    correct_num=0
                    self.accuracy_log['train'].append(accu)
                    print("\nEpoch {}, train loss = {}, train accuracy = {}".format(epoch+1, loss, accu))

                    print("Testing: ", end='')
                    actual_test_length, test_loss, test_accuracy = self.test(test_length)
                    self.loss_log['test'].append(test_loss)
                    self.accuracy_log['test'].append(test_accuracy)
                    print("test loss = {}, test accuracy = {}".format(test_loss, test_accuracy))

                    fig, grad_chart = self.visualize()
                    plt.close(fig)
                    plt.close(grad_chart)

                    with open(self.dir+'/Reg_process_record.txt', 'w') as file:
                        file.write("time={}\n".format(time.time()))
                        file.write("Epoch {}/{}, {}/{}\n".format(epoch+1, Epochs, i+1, length))
                        file.write("train_loss={}\n".format(self.loss_log['train']))
                        file.write("test_loss={}\n".format(self.loss_log['test']))
                        file.write("train_accuracy={}\n".format(self.accuracy_log['train']))
                        file.write("test_accuracy={}\n".format(self.accuracy_log['test']))
                        file.write("avg_grad={}\n".format(self.avg_grad))

                    early_stop = False

                    if test_accuracy >= early_stop_threshold_v1:
                        print("Additional Testing Starts.")
                        with open(self.dir+'/Additional_test.txt', 'w') as AT_file:
                            AT_file.write("AT_L0: test_length={}, loss={}, accuracy={}\n".format(actual_test_length, test_loss, test_accuracy))
                            
                            actual_test_length, test_loss, L1_accuracy = self.test(640)
                            print("AT_L1 : accuracy = {}".format(L1_accuracy))
                            AT_file.write("AT_L1: test_length={}, loss={}, accuracy={}, pass={}\n".format(actual_test_length, test_loss, L1_accuracy, test_accuracy>=early_stop_threshold_v2))
                            
                            if L1_accuracy >= early_stop_threshold_v2:
                                actual_test_length, test_loss, L2_accuracy = self.test(1280)
                                print("AT_L2 Passed: accuracy = {}".format(L2_accuracy))
                                AT_file.write("AT_L2: test_length={}, loss={}, accuracy={}, pass={}\n".format(actual_test_length, test_loss, L2_accuracy, test_accuracy>=early_stop_threshold_v2))

                                if L2_accuracy >= early_stop_threshold_v2:
                                    early_stop = True
                                    AT_F = (L1_accuracy * 640 + L2_accuracy * 1280) / (640 + 1280)
                                    print("Match the condition for early stop. AT_F={}".format(AT_F))
                                    AT_file.write("AT_F={}\n".format(AT_F))

                            AT_file.write("AT finished.\n")

                if early_stop:
                    break  # Then finish the corrent epoch.

            if early_stop:
                break  # Then finish the whole training process.

        print("RegExe: training finished.")


    def load(self, load_unsup=True):
        '''
        Used to load the state dict of the regression model, as well as STDP model.
        load_unsup: whether the state dict of unsupervised SNN model should be loaded at the same time.
        '''
        loadmap='cpu'
        if torch.cuda.is_available()==True:
            loadmap='cuda'
        self.model.load_state_dict(torch.load(self.dir+'/regression.pt', map_location=loadmap))
        print("RegExe: classsifier model loaded.")
        if load_unsup==True:
            self.unsup_exe.load()


    def visualize(self, save=True) -> Tuple[matplotlib.figure.Figure]:
        '''
        Create statistics charts including:
        1. reg_result_chart (reg_result.jpg): including all the loss and accuracy.
        2. grad_chart (grad.jpg): parameter.abs().mean() for each trainable tensor in the classifier model.
        '''
        plt.rcParams['axes.unicode_minus']=False

        reg_result_chart=plt.figure(figsize=(8,16))
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
            plt.plot([x[i] for x in self.avg_grad], label="tensor {}".format(i+1))
        plt.legend()
        plt.tight_layout()

        if save==True:
            reg_result_chart.savefig(self.dir+'/reg_result.jpg')
            grad_chart.savefig(self.dir+'/grad.jpg')
            print("RegExe: visualization chart saved.")

        return reg_result_chart, grad_chart
