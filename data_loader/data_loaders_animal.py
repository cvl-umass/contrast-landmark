import numpy as np
import pandas as pd
import os
from PIL import Image
import glob
import torch
import torchfile
from os.path import join as pjoin
from utils.util import label_colormap
from utils.util import pad_and_crop
from scipy.io import loadmat
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
from data_loader.augmentations import get_composed_augmentations
import torch.nn.functional as F
import pickle

from io import BytesIO
import sys
from pathlib import Path
import matplotlib

class PcaAug(object):
    _eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
    _eigvec = torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def __call__(self, im):
        alpha = torch.randn(3) * self.alpha
        rgb = (self._eigvec * alpha.expand(3, 3) * self._eigval.expand(3, 3)).sum(1)
        return im + rgb.reshape(3, 1, 1)


class JPEGNoise(object):
    def __init__(self, low=30, high=99):
        self.low = low
        self.high = high

    def __call__(self, im):
        H = im.height
        W = im.width
        rW = max(int(0.8 * W), int(W * (1 + 0.5 * torch.randn([]))))
        im = TF.resize(im, (rW, rW))
        buf = BytesIO()
        im.save(buf, format='JPEG', quality=torch.randint(self.low, self.high,
                                                          []).item())
        im = Image.open(buf)
        im = TF.resize(im, (H, W))
        return im


def kp_normalize(H, W, kp):
    kp = kp.clone()
    kp[..., 0] = 2. * kp[..., 0] / (W - 1) - 1
    kp[..., 1] = 2. * kp[..., 1] / (H - 1) - 1
    return kp

def kp_unnormalize(H, W, kp):
    kp = kp.clone()
    kp[..., 0] = W * kp[..., 0]
    kp[..., 1] = H * kp[..., 1]
    return kp


class AnimalBase(Dataset):
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        im = Image.open(self.filenames[index]).convert("RGB") 
        kp = -1
        kp_normalized = -1 # None
        visible = -1
        if self.pair_image: # unsupervised contrastive learning 
            # randomresizecrop is the key to generate pairs of images
            img1 = self.transforms(self.initial_transforms(im))
            img2 = self.transforms(self.initial_transforms(im))
            data = torch.cat([img1, img2], dim=0)
            if self.crop != 0: # maybe useful for datasets other than celebA/MAFL
                data = data[:, self.crop:-self.crop, self.crop:-self.crop]
        else: # supervised postprocessing
            kp = self.keypoints[index].copy()
            data = self.transforms(self.initial_transforms(im))
            if self.crop != 0: # maybe useful for datasets other than celebA/MAFL
                data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                kp = kp - self.crop
            kp = torch.as_tensor(kp)
            C, H, W = data.shape
            # import pdb; pdb.set_trace()
            kp = kp_unnormalize(H, W, kp) # the initial form of kp is normalized to [0,1]
            kp_normalized = kp_normalize(H, W, kp)
            visible = self.visible[index]

        if self.visualize:
            # from torchvision.utils import make_grid
            from utils.visualization import norm_range
            plt.clf()
            fig = plt.figure()
            if self.pair_image:
                im1, im2 = torch.split(data, [3, 3], dim=0)
                ax = fig.add_subplot(121)
                ax.imshow(norm_range(im1).permute(1, 2, 0).cpu().numpy())
                ax = fig.add_subplot(122)
                ax.imshow(norm_range(im2).permute(1, 2, 0).cpu().numpy())
                print(im1.shape, im2.shape)
            else:
                ax = fig.add_subplot(111)
                ax.imshow(norm_range(data).permute(1, 2, 0).cpu().numpy())
                kp_x = kp[visible][:, 0].numpy()
                kp_y = kp[visible][:, 1].numpy()
                ax.scatter(kp_x, kp_y)
                print(data.shape)
            # plt.savefig('check_dataloader.png', bbox_inches='tight')
            plt.savefig(os.path.join('sanity_check', vis_name + '.png'), bbox_inches='tight')
            print(self.filenames[index])  
            plt.close()
        # import pdb; pdb.set_trace()
        return data, visible, kp_normalized, index


class InatAve(AnimalBase):

    def __init__(self, root, train=True, pair_image=True, imwidth=224, crop=0,
                 do_augmentations=True, visualize=False, imagelist=None, **kwargs):

        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.pair_image = pair_image
        self.visualize = visualize
        self.crop = crop

        # get the imagelist
        if imagelist is not None:
            print('Load data from %s' % imagelist)
            with open(imagelist, 'r') as f:
                self.filenames =  [x.strip() for x in f]
        else:
            print('Load data from %s' % self.root)
            self.filenames = glob.glob(os.path.join(self.root, '*', '*jpg'))

        print('Number of images from Inat Ave.: %d' % len(self.filenames))

        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                         std  = [0.229, 0.224, 0.225])
        augmentations = [
            JPEGNoise(),
            # the only augmentation added on the top of DVE
            transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Resize((self.imwidth, self.imwidth))
        self.transforms = transforms.Compose(augmentations + [normalize])

    def __len__(self):
        return len(self.filenames)


class CUB(AnimalBase):
    # place the annotations file under /datasets/CUB-200-2011/anno
    # place the train/val/test text files under /datasets/CUB-200-2011/split

    def load_annos(self):
        train_annos = torchfile.load(os.path.join(self.root, 'anno', 'train.dat'))
        val_annos = torchfile.load(os.path.join(self.root, 'anno', 'val.dat'))
        train_val_annos = {**train_annos, **val_annos}
        annos = {}
        for name, kp in train_val_annos.items():
            name = name.decode()
            annos[name] = {}
            for idx, loc in kp.items():
                annos[name][int(idx.decode())] = tuple(loc)
        return annos

    def load_txt(self, imagetxt):
        return [line.rstrip('\n') for line in open(imagetxt)]


    def __init__(self, root, train=False, val=False, test=False, imagelist=None,
                 pair_image=False, imwidth=224, crop=0, visualize=False, **kwargs):

        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.val   = val
        self.test  = test
        self.pair_image = pair_image
        self.visualize = visualize
        self.crop = crop
        self.kp_num = 15

        # load training/val/test txt
        #train_path = os.path.join(root, 'split', 'train.txt')
        val_path   = os.path.join(root, 'split',  'val.txt')
        test_path  = os.path.join(root, 'split',  'test.txt')

        # get the imagelist
        annos = self.load_annos()
        if train:
            prefix = 'Train'
            print('Load image from %s' % imagelist)
            self.filenames = self.load_txt(imagelist)
        elif val:
            prefix = 'Val'
            print('Load image from %s' % val_path)
            self.filenames = self.load_txt(val_path)
        elif test:
            prefix = 'Test'
            print('Load image from %s' % test_path)
            self.filenames = self.load_txt(test_path)
            
        # double check the format of the loaded annotations
        self.keypoints = []
        self.visible = []
        for fname in self.filenames:
            keypoints = []
            visible = []
            kps = annos[fname.split('/')[-1]] # change the format of keypoints
            for idx in range(self.kp_num):
                if int(kps[idx+1][2]) == 1: # the keyvalue is from 1 to 15
                    keypoints.append([kps[idx+1][0], kps[idx+1][1]])
                    visible.append(True)
                else:
                    keypoints.append([0, 0])
                    visible.append(False)
            self.keypoints.append(np.array(keypoints))
            self.visible.append(np.array(visible))

        print('%s: number of images: %d; number of keypoints: %d' % (prefix, len(self.filenames), len(self.keypoints)))

        normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                         std  = [0.229, 0.224, 0.225])
        self.initial_transforms = transforms.Resize((self.imwidth, self.imwidth))
        self.transforms = transforms.Compose([transforms.ToTensor(), normalize])

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Helen")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--pair_image", action="store_true")
    parser.add_argument("--do_augmentations", action="store_true")
    parser.add_argument("--vis_name", type=str, default='check_dataloader')
    parser.add_argument("--imagelist", type=str, default='list of image to load')
    args = parser.parse_args()

    imwidth = 96
    crop = 0

    vis_name = args.vis_name
    kwargs = {
        "train": args.train,
        "val": args.val,
        "test": args.test,
        "pair_image": args.pair_image,
        'do_augmentations': args.do_augmentations,
        'vis_name': args.vis_name,
        "visualize": True,
        "imwidth": imwidth,
        "crop": crop,
        'imagelist': args.imagelist
    }

    dataset = globals()[args.dataset](**kwargs)
    dataset[0]
