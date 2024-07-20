'''
分类器执行器RegExe的装配方法：
    1. nn.Module(STDP模型) -> STDPExe(STDP模型执行器)
    2. STDPExe + nn.Module(分类器模型) -> RegExe
'''

import torch
import SNN_pkg as pkg

DEAP_dir="/home/featurize/work/Project/DEAP"
train_data, test_data = pkg.tools.SimpleDeap(DEAP_dir, mode="spiking", memory_num=10).split(0.1)
print("test length: {}".format(len(test_data)))
train_ds=torch.utils.data.DataLoader(train_data, batch_size=8)
test_ds=torch.utils.data.DataLoader(test_data, batch_size=32)

EEG_model=pkg.StdpModel.EEG_SequentialCompressionUnit("spiking")
STDP_exe=pkg.StdpModel.STDPExe(EEG_model, "model", train_ds, test_ds)

DNN=pkg.Regression.EEG_DNN() # 这里用含两个隐藏层的神经网络作分类器，逻辑回归用EEG_Reg类
DNN_exe=pkg.Regression.RegExe(DNN, STDP_exe, "model", train_ds, test_ds) # 示例化时自动从model文件夹里加载STDP模型

DNN_exe.train(1)
