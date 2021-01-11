import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
matplotlib.use('Agg')

from lib.data import makeCatsDataset
from lib.networks import Generator, Discriminator, Discriminator_WGAN, Progressive_Discriminator, Progressive_Generator, weights_init

from lib.misc import noisy, image_with_title
from lib.model.DCGAN import DCGAN

import math
import numpy as np
from tqdm import tqdm
import os

class Progressive_DCGAN(DCGAN):
    def __init__(self, opt):
        self.exp_name = opt.exp_name
        self.batch = opt.batch
        self.latent = opt.latent
        self.isize = opt.isize
        self.cur_isize = 4
        self.device = opt.device
        self.data_path = opt.data_path

        self.epochs = opt.epochs
        self.lr_d = opt.lr_d
        self.lr_g = opt.lr_g
        self.lr_decay_epoch = opt.lr_decay_epoch
        self.lr_decay_factor = opt.lr_decay_factor
        self.g_it = opt.g_it
        self.d_it = opt.d_it
        self.b1 = opt.b1
        self.b2 = opt.b2
        self.noise = opt.noise
        self.lambda_coff = opt.lambda_coff
        

        self.dataloader = makeCatsDataset(path=self.data_path, batch=self.batch, isize=self.cur_isize)
        self.gen = Progressive_Generator(self.latent, device=self.device)
        # self.gen.add_block()
        # self.gen.end_transition()
        
        self.dis = Progressive_Discriminator(device=self.device)
        # self.dis.add_block()
        # self.dis.end_transition()

        # self.gen.apply(weights_init)
        # self.dis.apply(weights_init)

    def setup_train(self):
        self.fixed_noise = torch.randn(36, self.latent, device=self.device)
        self.fixed_noise_64 = torch.randn(64, self.latent, device=self.device)
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.real_label = 0.9
        self.fake_label = 0.0

        self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(self.b1, self.b2))
        self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)) 
        # self.criterion = nn.BCELoss()
        self.criterion = nn.MSELoss()

    def train(self):
        self.save_folder = os.path.join('./out', self.exp_name + '/')
        self.init_folder()

        print("Strated {}\nepochs: {}\ndevice: {}".format(self.exp_name, self.epochs, self.device))
        
        self.setup_train()

        self.pbar = tqdm()

        alpha_inc = 1.0 / (self.epochs-1)

        while self.cur_isize < self.isize:
            print(f"train {self.cur_isize}x{self.cur_isize}")
            self.transition = False
            self.pbar.reset(total=self.epochs*len(self.dataloader))
            for epoch in range(self.epochs):
                self.train_one_epoch()
                self.save_progress_image()
            
            self.transition = True
            self.gen.add_block()
            self.dis.add_block()
            self.cur_isize *= 2
            self.dataloader = makeCatsDataset(path=self.data_path, batch=self.batch, isize=self.cur_isize)
            self.alpha = alpha_inc

            print(f"train transition {self.cur_isize}x{self.cur_isize}")
            self.pbar.reset(total=self.epochs*len(self.dataloader))
            for epoch in range(self.epochs):
                self.train_one_epoch()
                self.save_progress_image()
                self.alpha += alpha_inc
            
            self.gen.end_transition()
            self.dis.end_transition()

            # self.make_chart()
            # self.save_weights()
        print(f"train {self.cur_isize}x{self.cur_isize}")
        self.pbar.reset(total=self.epochs*len(self.dataloader))
        self.transition = False
        for epoch in range(self.epochs):
            self.train_one_epoch()
            self.save_progress_image()


    def train_discriminator(self):
        self.op_dis.zero_grad()
        # True
        imgs = self.data_device

        if self.transition:
            output_real = self.dis.transition_forward(imgs, self.alpha).view(-1)
        else:
            output_real = self.dis(imgs).view(-1)

        label_real = torch.full((output_real.size()[0],), self.real_label, device=self.device) 
        self.D_x = output_real.mean().item()

        # False
        z = torch.randn(imgs.size()[0], self.latent, device=self.device)
        if self.transition:
            imgs = self.gen.transition_forward(z, self.alpha)
        else:
            imgs = self.gen(z)
        # add noise on image    
        if self.noise:
            imgs = noisy(imgs, self.device)

        if self.transition:
            output_fake = self.dis.transition_forward(imgs, self.alpha).view(-1)
        else:
            output_fake = self.dis(imgs).view(-1)

        label_fake = torch.full((output_fake.size()[0],), self.fake_label, device=self.device)

        real_loss = self.criterion(output_real, label_real)
        fake_loss = self.criterion(output_fake, label_fake)

        self.d_loss = (real_loss + fake_loss) / 2
        self.d_loss.backward()

        self.D_G_z1 = output_fake.mean().item()

        self.op_dis.step()

    def train_generator(self):
        self.op_gen.zero_grad()

        z = torch.randn(self.data_device.size()[0], self.latent, device=self.device)

        if self.transition:
            imgs = self.gen.transition_forward(z, self.alpha)
        else:
            imgs = self.gen(z)

        if self.noise:
            imgs = noisy(imgs, self.device)

        if self.transition:
            output_fake = self.dis.transition_forward(imgs, self.alpha).view(-1)
        else:
            output_fake = self.dis(imgs).view(-1)

        label_g = torch.full((output_fake.size()[0],), self.real_label, device=self.device) 
        
        self.g_loss = self.criterion(output_fake, label_g)
        self.g_loss.backward()

        self.op_gen.step()
        self.D_G_z2 = output_fake.mean().item()

    def save_progress_image(self):
        with torch.no_grad():
            if self.transition:
                fake = self.gen.transition_forward(self.fixed_noise_64, self.alpha).detach().cpu()
            else:
                fake = self.gen(self.fixed_noise_64).detach().cpu()
        name = "final_{}x{}".format(self.cur_isize, self.cur_isize)
        if self.transition:
            name += "_transition"
        name += ".png"

        vutils.save_image(fake, self.save_folder + name, normalize=True)
