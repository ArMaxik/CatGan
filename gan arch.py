class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.l1 = nn.Linear(in_features=LATENT, out_features=1024*4*4)
        self.bl1 = nn.BatchNorm2d(1024)

        self.c1 = nn.ConvTranspose2d(1024, 512, (4, 4), 2, 1, bias=False) # ->8x8
        self.b1 = nn.BatchNorm2d(512)

        self.c2 = nn.ConvTranspose2d(512, 512, (3, 3), 1, 1, bias=False) # ->8x8
        self.b2 = nn.BatchNorm2d(512)

        self.c3 = nn.ConvTranspose2d(512, 256, (4, 4), 2, 1, bias=False) # ->16x16
        self.b3 = nn.BatchNorm2d(256)

        self.c4 = nn.ConvTranspose2d(256, 256, (3, 3), 1, 1, bias=False) # ->16x16
        self.b4 = nn.BatchNorm2d(256)

        self.c5 = nn.ConvTranspose2d(256, 128, (4, 4), 2, 1, bias=False) # ->32x32
        self.b5 = nn.BatchNorm2d(128)
        
        self.c6 = nn.ConvTranspose2d(128, 64, (3, 3), 1, 1, bias=False) # ->16x16
        self.b6 = nn.BatchNorm2d(64)

        self.c7 = nn.ConvTranspose2d(64, 3, (4, 4), 2, 1, bias=False) # ->64x64

        self.drop = nn.Dropout(p=0.3)

    def forward(self, z1, z2):
        x = self.l1(z1)
        x = self.drop(x).view(-1, 1024, 4, 4)
        x = self.bl1(x)
        x = F.leaky_relu(x)

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
        
        x = self.c6(x)
        x = self.b6(x)
        x = F.leaky_relu(x)
        
        x = self.c7(x)
        

        x = torch.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.c1 = nn.Conv2d(3, 64, 3, 2, padding=1)  # ->32x32
        self.b1 = nn.BatchNorm2d(64)
        
        self.c2 = nn.Conv2d(64, 64, 3, 1, padding=1)  # ->32x32
        self.b2 = nn.BatchNorm2d(64)

        self.c3 = nn.Conv2d(64, 128, 3, 2, padding=1)  # ->16x16
        self.b3 = nn.BatchNorm2d(128)
        
        self.c4 = nn.Conv2d(128, 128, 3, 1, padding=1)  # ->16x16
        self.b4 = nn.BatchNorm2d(128)
        

        self.c5 = nn.Conv2d(128, 64, 3, 2, padding=1)  # ->8x8
        self.b5 = nn.BatchNorm2d(64)
        
        self.c6 = nn.Conv2d(64, 32, 3, 2, padding=1)  # ->4x4
        self.b6 = nn.BatchNorm2d(32)
        

        self.l1 = nn.Linear(in_features=32*4*4, out_features=8*4*4)
        self.l2 = nn.Linear(in_features=8*4*4, out_features=1)

        
        self.drop = nn.Dropout(p=0.3)
        
    def forward(self, z1):
        x = self.c1(z1)
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
        
        x = self.c6(x)
        x = self.b6(x)
        x = F.leaky_relu(x)
        
        x = x.view(-1, 32*4*4)

        x = self.l1(x)
        x = self.drop(x)
        x = F.leaky_relu(x)

        
        x = self.l2(x)

        return torch.sigmoid(x)

# ====== 1 ====== 1 ====== 1 ======