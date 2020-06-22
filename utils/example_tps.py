import tps
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pdb
import numpy as np
import torch

def draw_points(im, kps):
    for i in range(kps.shape[0]):
        y = int(kps[i, 0])
        x = int(kps[i, 1])
        im[:, x-3:x+3, y-3:y+3] = 0
    return im

imwidth = 300
warper = tps.WarperSingle(H=imwidth, W=imwidth)
im = Image.open('test.jpg').convert("RGB")
initial_transforms = transforms.Compose([transforms.Resize((imwidth, imwidth))])
im1 = initial_transforms(im)
im1 = TF.to_tensor(im1) * 255
kp1 = np.random.randint(100, imwidth-100, size=(10,2))
im2, kp2 = warper(im1, keypts=kp1)

im2 = im2.to(torch.uint8)
im2 = draw_points(im2, kp2)
im2 = TF.to_pil_image(im2)
im1 = im1.to(torch.uint8)
im1 = draw_points(im1, kp1)
im1 = TF.to_pil_image(im1)

im1.save('test2.jpg')
im2.save('test3.jpg')

print(kp1) 
print(kp2)
