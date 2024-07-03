import torch
import SNN_pkg as pkg

data=pkg.tools.SimpleDeap("/home/featurize/work/Project/DEAP",1)
ds=torch.utils.data.DataLoader(data, batch_size=1)
EEG_model=pkg.StdpModel.EEG_STDPModel()
exe=pkg.StdpModel.STDPExe(EEG_model, "model", ds, ds)
exe.load()
exe.test()
