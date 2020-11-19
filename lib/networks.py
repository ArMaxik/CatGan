import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, LATENT):
        super(Generator, self).__init__()

        self.z = LATENT

        self.l1 = nn.Linear(in_features=self.z, out_features=self.z*2)
        self.lb1 = nn.BatchNorm1d(self.z*2)
        self.l2 = nn.Linear(in_features=self.z*2, out_features=self.z*3)
        self.lb2 = nn.BatchNorm1d(self.z*3)
        self.l3 = nn.Linear(in_features=self.z*3, out_features=self.z)
        self.lb3 = nn.BatchNorm1d(self.z)

        self.c1 = nn.ConvTranspose2d(self.z, 512, (4, 4), 1, 0, bias=True) # -> 4x4
        self.b1 = nn.BatchNorm2d(512)

        self.c2 = nn.ConvTranspose2d(512,    128, (4, 4), 2, 1, bias=True) # -> 8x8
        self.b2 = nn.BatchNorm2d(128)

        self.c3 = nn.ConvTranspose2d(128,    64, (4, 4), 2, 1, bias=True) # -> 16x16
        self.b3 = nn.BatchNorm2d(64)
        
#         self.cc1 = nn.Conv2d(512, 256, (3, 3), 1, 1, bias=True) # -> 16x16
#         self.ccb1 = nn.BatchNorm2d(256)
        
        self.c4 = nn.ConvTranspose2d(64,     64, (4, 4), 2, 1, bias=True) # -> 32x32
        self.b4 = nn.BatchNorm2d(64)
        
#         self.cc2 = nn.Conv2d(256, 3, (3, 3), 1, 1, bias=True) # -> 32x32
#         self.ccb2 = nn.BatchNorm2d(128)

        
        self.c5 = nn.ConvTranspose2d(64,      3, (4, 4), 2, 1, bias=True) # -> 64x64
        self.b5 = nn.BatchNorm2d(3)
        
#         self.cc3 = nn.Conv2d(64, 3, (3, 3), 1, 1, bias=True) # -> 64x64

        self.drop = nn.Dropout(p=0.01)

    def forward(self, z1):
        x = self.l1(z1)
        x = self.lb1(x)
        x = F.leaky_relu(x)
        
        x = self.l2(x)
        x = self.lb2(x)
        x = F.leaky_relu(x)
        
        x = self.l3(x)
        x = self.lb3(x)
        x = F.leaky_relu(x)
        
        x = x.view(-1, self.z, 1, 1)
    
        x = self.c1(x)
        x = self.b1(x)
        x = F.leaky_relu(x)

        x = self.c2(x)
        x = self.b2(x)
        x = F.leaky_relu(x)

        x = self.c3(x)
        x = self.b3(x)
        x = F.leaky_relu(x)
        
#         x = self.cc1(x)
#         x = self.ccb1(x)
#         x = F.leaky_relu(x)

        x = self.c4(x)
        x = self.b4(x)
        x = F.leaky_relu(x)
                
#         x = self.cc2(x)
#         x = self.ccb2(x)
#         x = F.leaky_relu(x)

        x = self.c5(x)
        x = self.b5(x)
#         x = F.leaky_relu(x)
        
#         x = self.cc3(x)
        x = torch.tanh(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.c1 = nn.Conv2d(3, 64, 4, 2, padding=1)  # ->32x32
        self.b1 = nn.BatchNorm2d(64)
        
        self.c2 = nn.Conv2d(64, 64, 3, 1, padding=1)  # ->32x32
        self.b2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(64, 128, 4, 2, padding=1)  # ->16x16
        self.b3 = nn.BatchNorm2d(128)
        
        self.c4 = nn.Conv2d(128, 256, 4, 2, padding=1)  # ->8x8
        self.b4 = nn.BatchNorm2d(256)
        
        self.c5 = nn.Conv2d(256, 512, 4, 2, padding=1)  # ->4x4
        self.b5 = nn.BatchNorm2d(512)
        

#         self.c5 = nn.Conv2d(512, 512, 3, 2, padding=1)  # ->8x8
#         self.b5 = nn.BatchNorm2d(64)
        
#         self.c6 = nn.Conv2d(512, 1024, 3, 2, padding=1)  # ->4x4
#         self.b6 = nn.BatchNorm2d(32)
        
#         self.c5 = nn.Conv2d(512, 512, 3, 1, padding=1)   # ->8x8

        self.l1 = nn.Linear(in_features=512*4*4, out_features=1)
#         self.l2 = nn.Linear(in_features=8*4*4, out_features=1)

        
        self.drop = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = F.leaky_relu(x)

        x = self.c2(x)
        x = self.b2(x)
        x = F.leaky_relu(x)

        x = self.c3(x)
        x = self.b3(x)
        x = F.leaky_relu(x)
        
        x = self.c4(x)
        x = self.b4(x)
        x = F.leaky_relu(x)
        
        x = self.c5(x)
        x = self.b5(x)
        x = F.leaky_relu(x)
        
#         x = self.c6(x)
#         x = self.b6(x)
#         x = F.leaky_relu(x)
        
        x = x.view(-1, 512*4*4)
        x = self.l1(x)

        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    