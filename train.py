from lib.model import CatGAN


gan = CatGAN()
CatGAN.EPOCHS = 250
CatGAN.EXP_NAME = "DCGAN_1"
gan.train()
 
