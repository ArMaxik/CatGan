from lib.model.Progressive_DCGAN import Progressive_DCGAN
from lib.model.Progressive_WGAN import Progressive_WGAN
from lib.model.WGAN import WGAN_GP

import torch

class options:
    def __init__(self):
        self.exp_name = "WGAN_05"
        self.batch = 256
        self.latent = 128
        self.isize = 128
        self.device = torch.device("cuda:1" if (torch.cuda.is_available()) else "cpu")
        self.device_ids = [1, 2]
        # self.data_path = "/home/v-eliseev/Datasets/cats/"
        # self.data_path = "/mnt/p/datasets/cats/"
        self.data_path = "/raid/veliseev/datasets/cats/"

        self.epochs = 3
        self.lr_d = 0.0004
        self.lr_g = 0.0004
        self.lr_decay_epoch = [1000]
        self.lr_decay_factor = 10.0
        self.g_it = 1
        self.d_it = 1
        self.b1 = 0.0
        self.b2 = 0.9
        self.noise = False
        self.lambda_coff = 10.0
    

opt = options()

gan = WGAN_GP(opt)
gan.train()

with open("./out/{}/opt.txt".format(opt.exp_name), 'w') as f:
    for (k, v) in opt.__dict__.items():
        f.write("{:24s}{}\n".format(k, v))
