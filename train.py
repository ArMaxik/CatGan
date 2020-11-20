from lib.model import CatGAN
import torch

class options:
    def __init__(self):
        self.exp_name = "DCGAN_bce_loss_lr-decay_noise"
        self.batch = 64
        self.latent = 100
        self.isize = 64
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.data_path = "/home/v-eliseev/Datasets/cats/"

        self.epochs = 100
        self.lr_d = 0.0002
        self.lr_g = 0.0002
        self.lr_decay_epoch = [50, 75]
        self.lr_decay_factor = 10.0
        self.noise = True

opt = options()

gan = CatGAN(opt)
gan.train()
 
