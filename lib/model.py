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
    def __init__(self, opt):
        self.exp_name = opt.exp_name
        self.batch = opt.batch
        self.latent = opt.latent
        self.isize = opt.isize
        self.device = opt.device
        self.data_path = opt.data_path

        self.epochs = opt.epochs
        self.lr_d = opt.lr_d
        self.lr_g = opt.lr_g

        self.dataloader = makeCatsDataset(path=self.data_path, batch=self.batch)
        self.gen = Generator(self.latent).to(self.device)
        self.dis = Discriminator().to(self.device)
        self.gen.apply(weights_init)
        self.dis.apply(weights_init)

    def train_discriminator(self):
        self.op_dis.zero_grad()
        # True

#             imgs = noisy(data_device, device=self.device)
        imgs = self.data_device

        output_real = self.dis(imgs).view(-1)

        label_real = torch.full((output_real.size()[0],), self.real_label, device=self.device) 
        self.D_x = output_real.mean().item()

        # False
        z = torch.randn(imgs.size()[0], self.latent, device=self.device)
        imgs = self.gen(z)
#             imgs = noisy(imgs, device=self.device)

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

#             imgs = noisy(data_device, device=device)
        imgs = self.data_device

        z = torch.randn(imgs.size()[0], self.latent, device=self.device)
        output_fake = self.dis(self.gen(z)).view(-1)

        label_g = torch.full((output_fake.size()[0],), self.real_label, device=self.device) 
        
        self.g_loss = self.criterion(output_fake, label_g)
        self.g_loss.backward()

        self.op_gen.step()
        self.D_G_z2 = output_fake.mean().item()
    
    def train_one_epoch(self):
        for i, data in enumerate(self.dataloader, 0):
            self.data_device = data.to(self.device)

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
        self.save_folder = os.path.join('./out', self.exp_name + '/')
        self.init_folder()

        print("Strated {}\nepochs: {}\ndevice: {}".format(self.exp_name, self.epochs, self.device))
        
        self.fixed_noise = torch.randn(36, self.latent, device=self.device)
        self.img_list = []
        self.G_losses = []
        self.D_losses = []
        self.real_label = 0.9
        self.fake_label = 0.0

        self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=(0.5, 0.999))
        self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=self.lr_d, betas=(0.5, 0.999)) 
        self.criterion = nn.BCELoss()
        # self.criterion = nn.BCEWithLogitsLoss()

        self.pbar = tqdm()
        self.pbar.reset(total=self.epochs*len(self.dataloader))  # initialise with new `total`

        for epoch in range(self.epochs):
            # if epoch == 25:
            #     self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=0.00001, betas=(0.5, 0.999))
            #     self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=0.00001, betas=(0.5, 0.999))
            # if epoch == 50:
            #     self.op_gen = torch.optim.Adam(self.gen.parameters(), lr=0.000001, betas=(0.5, 0.999))
            #     self.op_dis = torch.optim.Adam(self.dis.parameters(), lr=0.000001, betas=(0.5, 0.999))
            
            self.train_one_epoch()
            self.make_stats()

            print('[{:3d}/{:d}] Loss_D: {:.4f}  Loss_G: {:.4f} | D(x): {:.4f}  D(G(z)): {:.4f} / {:.4f}'.format(
                epoch, self.epochs-1,
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

        noise = torch.randn(64, self.latent, device=self.device)
        with torch.no_grad():
            fake = self.gen(noise).detach().cpu()
        vutils.save_image(fake, self.save_folder + "final.png", normalize=True)

    def save_video(self):
        fig = plt.figure(figsize=(12,12))
        ims = [
            image_with_title(img,
                            "Epoch: {}".format(i),
                            "[RGAN] batch size: {0}, latent space: {1}, size {2}x{2}".format(self.batch, self.latent, 32))
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