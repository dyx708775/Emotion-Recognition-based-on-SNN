import SNN_StdpModel as Stdp
import SNN_Regression as Reg
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor,ToPILImage
import torch
import sys

writer = torch.utils.tensorboard.SummaryWriter('/home/xiangnan/Files/code/SNN/Emotion-Recognition-based-on-SNN/Network/src_files')

mnist_dir='/home/xiangnan/Files/code/mnist'
stdpmod_dir='/home/xiangnan/Files/code/SNN/Emotion-Recognition-based-on-SNN/Network/model/STDP'
regmod_dir='/home/xiangnan/Files/code/SNN/Emotion-Recognition-based-on-SNN/Network/model/Regression'

train_ds=MNIST(
    root=mnist_dir,
    download=True,
    transform=ToTensor(),
    train=True)
test_ds=MNIST(
    root=mnist_dir,
    download=True,
    transform=ToTensor(),
    train=False)
class SmallDataset(torch.utils.data.Dataset):
    def __init__(self, ds, start: int, end: int):
        self.ds=ds
        self.range=(start,end)
    def __len__(self):
        x,y=self.range
        return y
    def __getitem__(self, i):
        if i>=len(self):
            raise StopIteration
        return self.ds[i]
train_data=SmallDataset(train_ds,0,2000)
test_data=SmallDataset(test_ds,0,150)

model=Stdp.STDPModel()
stdp_exe=Stdp.STDPExe(model, stdpmod_dir, train_data, test_ds)

x,y=test_data[0]
writer.add_graph(model, x)
writer.close()
sys.exit()

reg_model=Reg.RegressionModel()
reg_exe=Reg.RegExe(reg_model, stdp_exe, regmod_dir, train_ds, test_data)
