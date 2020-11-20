import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, LATENT):
        super(Generator, self).__init__()

        self.z = LATENT

        self.c1 = nn.ConvTranspose2d(self.z, 512, (4, 4), 1, 0, bias=True) # -> 4x4
        self.b1 = nn.BatchNorm2d(512)

        self.c2 = nn.ConvTranspose2d(512,    256, (4, 4), 2, 1, bias=True) # -> 8x8
        self.b2 = nn.BatchNorm2d(256)

        self.c3 = nn.ConvTranspose2d(256,    128, (4, 4), 2, 1, bias=True) # -> 16x16
        self.b3 = nn.BatchNorm2d(128)
        
        self.c4 = nn.ConvTranspose2d(128,     64, (4, 4), 2, 1, bias=True) # -> 32x32
        self.b4 = nn.BatchNorm2d(64)
        
        self.c5 = nn.ConvTranspose2d(64,      3, (4, 4), 2, 1, bias=True) # -> 64x64

        self.drop = nn.Dropout(p=0.01)

    def forward(self, x):
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
        
        x = self.c4(x)
        x = self.b4(x)
        x = F.leaky_relu(x)
                
        x = self.c5(x)

        x = torch.tanh(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.c1 = nn.Conv2d(3, 64, 4, 2, padding=1)  # ->32x32
        self.b1 = nn.BatchNorm2d(64)
        
        self.c2 = nn.Conv2d(64, 128, 3, 1, padding=1)  # ->32x32
        self.b2 = nn.BatchNorm2d(128)

        self.c3 = nn.Conv2d(128, 256, 4, 2, padding=1)  # ->16x16
        self.b3 = nn.BatchNorm2d(256)
        
        self.c4 = nn.Conv2d(256, 512, 4, 2, padding=1)  # ->8x8
        self.b4 = nn.BatchNorm2d(512)
        
        self.c5 = nn.Conv2d(512, 1, 4, 2, padding=1)  # ->4x4
        

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
        # x = torch.sigmoid(x)
        
        
        # x = x.view(-1, 512*4*4)

        return x.view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    