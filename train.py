from lib.model import DCGAN, WGAN_GP
import torch

class options:
    def __init__(self):
        self.exp_name = "WGAN-GP_2"
        self.batch = 64
        self.latent = 100
        self.isize = 64
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        # self.data_path = "/home/v-eliseev/Datasets/cats/"
        self.data_path = "/mnt/p/datasets/cats/"

        self.epochs = 250
        self.lr_d = 0.0001
        self.lr_g = 0.0001
        self.lr_decay_epoch = [200, 225]
        self.lr_decay_factor = 10.0
        self.g_it = 1
        self.d_it = 5
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
