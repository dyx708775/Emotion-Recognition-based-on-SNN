#!/home/featurize/work/snn/bin/python

import torch
import SNN_pkg as pkg

DEAP_dir="/home/featurize/data/ZZH_code_2"
basic_data = pkg.tools.SimpleDeap(DEAP_dir, mode="zzh", memory_num = 500)
train_data, valid_data, test_data = basic_data.split(0.1)
print("test length: {}".format(len(test_data)))
train_ds=torch.utils.data.DataLoader(train_data, batch_size=16)
test_ds=torch.utils.data.DataLoader(valid_data, batch_size=64)

lsm_model=pkg.StdpModel.EEG_LSM()
lsm_exe = pkg.StdpModel.STDPExe(lsm_model, "Reg_model",  train_ds, test_ds)

dnn = pkg.Regression.EEG_Reg()
dnn_exe = pkg.Regression.RegExe(dnn, lsm_exe, "Reg_model", train_ds, test_ds)
dnn_exe.train(1, early_stop_accuracy = (0.48, 0.44), test_length = 256)

