import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.utils as vutils

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from lib.data import makeCatsDataset
from lib.networks import Generator, Discriminator, weights_init

import math
import numpy as np
from tqdm import tqdm
import os

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def image_with_title(img, title_text, info_text):
    plt.axis('off')
    title = plt.text(0,-7,
                    title_text, 
                    fontsize=26)
    title.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    # info = plt.text(0,32*6+22,
    #                 info_text, 
    #                 fontsize=14)
    # info.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    img_n = plt.imshow(np.transpose(img,(1,2,0)), animated=True)
    return [img_n, title]

class CatGAN:
    def __init__(self):
        self.EXP_NAME = "CatGAN_1"
        self.BATCH = 64
        self.LATENT = 100
        self.ISIZE = 64
        self.DEVICE = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

        self.EPOCHS = 4

        self.dataloader = makeCatsDataset(path='/mnt/p/datasets/cats/', batch=self.BATCH)
        self.gen = Generator(self.LATENT).to(self.DEVICE)
        self.dis = Discriminator().to(self.DEVICE)
        self.gen.apply(weights_init)
        self.dis.apply(weights_init)

        self.save_folder = os.path.join('./out', self.EXP_NAME + '/')
        self.init_folder()

    def train_discriminator(self):
        self.op_dis.zero_grad()
        # True

#             imgs = noisy(data_device, device=self.DEVICE)
        imgs = self.data_device

        output_real = self.dis(imgs).view(-1)

        label_real = torch.full((output_real.size()[0],), self.real_label, device=self.DEVICE) 
        self.D_x = output_real.mean().item()

        # False
        z = torch.randn(imgs.size()[0], self.LATENT, device=self.DEVICE)
        imgs = self.gen(z)
#             imgs = noisy(imgs, device=self.DEVICE)

        output_fake = self.dis(imgs).view(-1)
        label_fake = torch.full((output_fake.size()[0],), self.fake_label, device=self.DEVICE)

        real_loss = self.criterion(output_real-output_fake, label_real)
        fake_loss = self.criterion(output_fake-output_real, label_fake)

        self.d_loss = (real_loss + fake_loss) / 2
        self.d_loss.backward()

        self.D_G_z1 = output_fake.mean().item()

        self.op_dis.step()

    def train_generator(self):
        self.op_gen.zero_grad()

#             imgs = noisy(data_device, device=device)
        imgs = self.data_device

        output_real = self.dis(imgs).view(-1)

        z = torch.randn(imgs.size()[0], self.LATENT, device=self.DEVICE)
        output_fake = self.dis(self.gen(z)).view(-1)

        label_g = torch.full((output_fake.size()[0],), self.real_label, device=self.DEVICE) 
        
        self.g_loss = self.criterion(output_fake-output_real, label_g)
        self.g_loss.backward()

        self.op_gen.step()
        self.D_G_z2 = output_fake.mean().item()
    
    def train_one_epoch(self):
        for i, data in enumerate(self.dataloader, 0):
            self.data_device = data.to(self.DEVICE)

            self.train_discriminator()
            self.train_generator()

            self.pbar.update()
            

    def make_stats(self):
        with torch.no_grad():
            fake = self.gen(self.fixed_noise).detach().cpu()
        self.img_list.append(vutils.make_grid(fake, padding=2, normalize=True, nrow=6))

        self.G_losses.append(self.g_loss.item())
        self.D_losses.append(self.d_loss.item())


    def train(self):
        self.fixed_noise = torch.randn(36, self.LATENT, device=self.DEVICE)
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.real_label = 0.9
        self.fake_label = 0.0

        self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=0.0001, betas=(0.5, 0.999)) 
        # criterion = nn.BCELoss()
        self.criterion = nn.BCEWithLogitsLoss()

        self.pbar = tqdm()
        self.pbar.reset(total=self.EPOCHS*len(self.dataloader))  # initialise with new `total`

        for epoch in range(self.EPOCHS):
            if epoch == 25:
                self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=0.00001, betas=(0.5, 0.999))
                self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=0.00001, betas=(0.5, 0.999))
            if epoch == 50:
                self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=0.000001, betas=(0.5, 0.999))
                self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=0.000001, betas=(0.5, 0.999))
            
            self.train_one_epoch()
            self.make_stats()

            print('[{:3d}/{:d}] Loss_D: {:.4f}  Loss_G: {:.4f} | D(x): {:.4f}  D(G(z)): {:.4f} / {:.4f}'.format(
                epoch, self.EPOCHS-1,
                self.d_loss.item(), self.g_loss.item(),
                sigmoid(self.D_x), sigmoid(self.D_G_z1), sigmoid(self.D_G_z2)
                ))
        self.make_chart()
        self.save_video()
        self.save_weights()

    def make_chart(self):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(self.G_losses,label="G")
        plt.plot(self.D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_folder + "losses.png")
        plt.close()

    def save_video(self):
        fig = plt.figure(figsize=(12,12))
        ims = [
            image_with_title(img,
                            "Epoch: {}".format(i),
                            "[RGAN] Batch size: {0}, Latent space: {1}, size {2}x{2}".format(self.BATCH, self.LATENT, 32))
            for i, img in enumerate(self.img_list)
            ]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(self.save_folder + 'hist.mp4', writer=writer)

    def save_weights(self):
        g_w = self.gen.state_dict()
        d_w = self.dis.state_dict()
        torch.save(g_w, self.save_folder + 'c_gen.pth')
        torch.save(d_w, self.save_folder + 'c_dis.pth')

    def init_folder(self):
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder)