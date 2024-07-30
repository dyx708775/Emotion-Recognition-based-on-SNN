#!/home/featurize/work/snn/bin/python

import torch
import SNN_pkg as pkg

DEAP_dir="/home/featurize/data/zzhcode"
basic_data = pkg.tools.SimpleDeap(DEAP_dir, mode="zzh")
train_data, valid_data, test_data = basic_data.split()
print("test length: {}".format(len(valid_data)))
train_ds=torch.utils.data.DataLoader(train_data, batch_size=1)
test_ds=torch.utils.data.DataLoader(valid_data, batch_size=1)

model=pkg.StdpModel.EEG_LSM()
lsm_exe = pkg.StdpModel.STDPExe(model, "STDP_model_zzh",  train_ds, test_ds)
lsm_exe.train()
