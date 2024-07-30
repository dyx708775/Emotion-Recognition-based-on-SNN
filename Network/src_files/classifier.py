#!/home/featurize/work/snn/bin/python

import torch
import SNN_pkg as pkg

DEAP_dir="/home/featurize/work/Project/BSA_code"
basic_data = pkg.tools.SimpleDeap(DEAP_dir, mode="BSA", memory_num=20, time = 6)
train_data, valid_data, test_data = basic_data.split()
print("test length: {}".format(len(test_data)))
train_ds=torch.utils.data.DataLoader(train_data, batch_size=32)
test_ds=torch.utils.data.DataLoader(valid_data, batch_size=128)

lsm_model=pkg.StdpModel.EEG_LSM()
lsm_exe = pkg.StdpModel.STDPExe(lsm_model, "Reg_model",  train_ds, test_ds)

dnn = pkg.Regression.EEG_DNN()
dnn_exe = pkg.Regression.RegExe(dnn, lsm_exe, "Reg_model", train_ds, test_ds)
dnn_exe.train(1, early_stop_accuracy = (0.48, 0.42), test_length = 256)

