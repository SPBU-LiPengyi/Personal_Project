import torch
from easydict import EasyDict as edict
cfgs = edict()

cfgs.batch_size = 128

# dataset
cfgs.num_class = 10
cfgs.image_shape = (32, 32)


# hyper
cfgs.num_epochs = 200
cfgs.learning_rate = 0.001

cfgs.start_schedule = 0.0001
cfgs.end_schedule = 0.02
cfgs.timesteps = 1000
cfgs.VERBOSE = False

# save model weight in how many epoch
cfgs.PRINT_FREQUENCY = 20

cfgs.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")