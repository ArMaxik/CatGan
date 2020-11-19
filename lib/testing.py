import torchvision
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import data
from networks import *

def imshow(img, name=None):
    fig, ax = plt.subplots()
    img = np.transpose(img.numpy(), (1, 2, 0))
    ax.imshow(img, interpolation='none')
    ax.axis('off')

    if name != None:
        fig.tight_layout()
        fig.savefig(name + ".png")
    else:
        fig.show()
    plt.close()

def image_with_title(img, title_text, info_text):
    plt.axis('off')
    title = plt.text(0,-7,
                    title_text, 
                    fontsize=26)
    title.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    info = plt.text(0,32*6+22,
                    info_text, 
                    fontsize=14)
    info.set_bbox(dict(facecolor='white', alpha=1.0, edgecolor='white'))
    img_n = plt.imshow(np.transpose(img,(1,2,0)), animated=True);
    return [img_n, title]

dataloader = data.makeCatsDataset(path='/mnt/p/datasets/cats/', batch=16)

img_list = []
for i_batch, im in enumerate(dataloader):
    im = (im+1.0)/2.0
    
    # imshow(torchvision.utils.make_grid(im, nrow=4), name=str(i_batch))
    img_list.append(torchvision.utils.make_grid(im, nrow=4))
    if i_batch == 25:
        break

fig = plt.figure(figsize=(12,12))
# fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
ims = [
    image_with_title(img,
                     "Epoch: {}".format(i),
                     "[RGAN] Batch size: {0}, Latent space: {1}, size {2}x{2}".format(16, 15, 32))
    for i, img in enumerate(img_list)
    ]
ani = animation.ArtistAnimation(fig, ims, interval=750, repeat_delay=1000, blit=True)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
ani.save('test.mp4', writer=writer)


LATENT = 100

gen = Generator(LATENT).cuda()
gen.apply(weights_init)

dis = Discriminator().cuda()
dis.apply(weights_init)

data = gen(torch.randn(16, LATENT).cuda()).cpu()
fig, ax = plt.subplots()
fig.dpi = 250
imshow(torchvision.utils.make_grid((data.detach()+1)/2, nrow=4), name='test')
print(dis(data.cuda()).detach())
plt.close()