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


class WGAN_GP(DCGAN):
    def __init__(self, opt):
        super().__init__(opt)

        self.device_ids = opt.device_ids
        self.dis = Discriminator_WGAN().to(self.device)

        if len(self.device_ids) > 1 and not (self.device == "cpu"):
            self.gen = nn.DataParallel(self.gen, device_ids=self.device_ids)
            self.dis = nn.DataParallel(self.dis, device_ids=self.device_ids)

        self.lambda_coff = opt.lambda_coff

    def gradien_penalty(self, imgs_real, imgs_fake):
        b, c, h, w = imgs_real.shape
        epsilon = torch.rand((b, 1, 1, 1), device=self.device).repeat(1, c, h, w)
        interpolate = epsilon*imgs_real + (1.0 - epsilon)*imgs_fake
        interpolate.requires_grad_(True)
        
        d_interpolate = self.dis(interpolate)

        gradients = torch.autograd.grad(
            outputs=d_interpolate,
            inputs=interpolate,
            grad_outputs=torch.ones(d_interpolate.shape, device=self.device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(b, -1)

        penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return self.lambda_coff*penalty

    def train_discriminator(self):
        self.op_dis.zero_grad()
        # True
        imgs_real = self.data_device

        output_real = self.dis(imgs_real).view(-1)

        self.D_x = output_real.mean().item()

        # False
        z = torch.randn(self.data_device.size()[0], self.latent, device=self.device)
        imgs_fake = self.gen(z)

        output_fake = self.dis(imgs_fake).view(-1)

        self.d_loss = output_fake.mean() - output_real.mean() + self.gradien_penalty(imgs_real, imgs_fake)
        self.d_loss.backward()

        self.D_G_z1 = output_fake.mean().item()

        self.op_dis.step()

    def train_generator(self):
        self.op_gen.zero_grad()

        z = torch.randn(self.data_device.size()[0], self.latent, device=self.device)

        imgs = self.gen(z)
        if self.noise:
            imgs = noisy(imgs, self.device)

        output_fake = self.dis(imgs).view(-1).mean()
        
        self.g_loss = -output_fake
        self.g_loss.backward()

        self.op_gen.step()
        self.D_G_z2 = output_fake.mean().item()

    def setup_train(self):
        self.fixed_noise = torch.randn(36, self.latent, device=self.device)
        self.fixed_noise_64 = torch.randn(64, self.latent, device=self.device)

        self.img_list = []
        self.G_losses = []
        self.D_losses = []

        self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(self.b1, self.b2))
        self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(self.b1, self.b2)) 
        self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()

    def save_progress_image(self):
        with torch.no_grad():
            fake = self.gen(self.fixed_noise_64).detach().cpu()
        name = "final.png"

        vutils.save_image(fake, self.save_folder + name, normalize=True)