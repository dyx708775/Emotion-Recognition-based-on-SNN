import torch
import SNN_pkg as pkg

data=pkg.tools.SimpleDeap("/home/featurize/work/Project/DEAP", mode="prep", memory_num=10)
train_data, test_data = data.split(0.)
print("test length: {}".format(len(test_data)))
train_ds=torch.utils.data.DataLoader(train_data, batch_size=1)
# test_ds=torch.utils.data.DataLoader(test_data, batch_size=16)
EEG_model=pkg.StdpModel.EEG_SequentialCompressionUnit("realnum")
STDP_exe=pkg.StdpModel.STDPExe(EEG_model, "model", train_ds, None)
STDP_exe.train(1)
