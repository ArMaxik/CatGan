from lib.model import CatGAN
import torch

class options:
    def __init__(self):
        self.exp_name = "DCGAN_std_loss"
        self.batch = 64
        self.latent = 100
        self.isize = 64
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.data_path = "/home/v-eliseev/Datasets/cats/"

        self.epochs = 100
        self.lr_d = 0.0002
        self.lr_g = 0.0002
opt = options()

gan = CatGAN(opt)
gan.train()
 
