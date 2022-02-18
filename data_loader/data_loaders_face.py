import numpy as np
import pandas as pd
import os
from PIL import Image
import glob
import torch
from os.path import join as pjoin
from utils.util import label_colormap
from utils.util import pad_and_crop
from utils import tps
from scipy.io import loadmat
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F

from io import BytesIO
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt 


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


class CelebABase(Dataset):

    def __len__(self):
        return len(self.filenames)

    def restrict_annos(self, num, datapath=None, outpath=None, repeat_flag=True, num_per_epoch=1000):

        if datapath is None:
            anno_count = len(self.filenames)
            pick = np.random.choice(anno_count, num, replace=False)
            # print(f"Picking annotation for images: {np.array(self.filenames)[pick].tolist()}")
            if repeat_flag:
                total_count = min(num_per_epoch, len(self.filenames))
                # total_count = len(self.filenames)
                assert num < total_count
                repeat = int(total_count // num)
                self.filenames = np.tile(np.array(self.filenames)[pick], repeat)
                self.keypoints = np.tile(self.keypoints[pick], (repeat, 1, 1))
            else:
                self.filenames = np.array(self.filenames)[pick]
                self.keypoints = self.keypoints[pick]

            with open(os.path.join(outpath, 'datalist.txt'), 'w') as f:
                f.writelines("%d\n" % idx for idx in pick)
            with open(os.path.join(outpath, 'namelist.txt'), 'w') as f:
                f.writelines("%s\n" % name for name in self.filenames)
        else:
            with open(os.path.join(datapath, 'datalist.txt'), 'r') as f:
                pick = np.array([int(line.rstrip()) for line in f.readlines()])
            if repeat_flag:
                total_count = min(num_per_epoch, len(self.filenames))
                # total_count = 1000 if len(self.filenames) > 1000 else len(self.filenames)
                assert num < total_count # only repeat if the number of images are too small
                repeat = int(total_count // num)
                self.filenames = np.tile(np.array(self.filenames)[pick], repeat)
                self.keypoints = np.tile(self.keypoints[pick], (repeat, 1, 1))
            else:
                self.filenames = np.array(self.filenames)[pick]
                self.keypoints = self.keypoints[pick]
            # for sanity check
            with open(os.path.join(datapath, 'namelist_resume.txt'), 'w') as f: 
                f.writelines("%s\n" % name for name in self.filenames)


    def __getitem__(self, index):
        im = Image.open(os.path.join(self.subdir, self.filenames[index])).convert("RGB")
        kp = -1
        kp_normalized = -1 # None
        # import pdb; pdb.set_trace()
        if self.pair_image: # unsupervised contrastive learning 
            if not self.TPS_aug:
                kp = self.keypoints[index].copy()
                # randomresizecrop is the key to generate pairs of images
                img1 = self.transforms(self.initial_transforms(im))
                img2 = self.transforms(self.initial_transforms(im))
                data = torch.cat([img1, img2], dim=0)
                if self.crop != 0: # maybe useful for datasets other than celebA/MAFL
                    data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                    kp = kp - self.crop
                # the following keypoints assuming there is not augmentation applied to images (random crops, resize etc.)
                kp1 = torch.tensor(kp)
                kp2 = kp1.clone()
                kp = torch.cat([kp1, kp2], 0)
                C, H, W = data.shape
                kp_normalized = torch.cat((kp_normalize(H, W, kp1), kp_normalize(H, W, kp2)), 0)
            else: 
                #  add TPS deformation for image matching, returns a pair of images, a.w.a keypoints  
                kp = self.keypoints[index].copy()
                im1 = self.initial_transforms(im)
                im1 = TF.to_tensor(im1) * 255
                im1, im2, flow, grid, kp1, kp2 = self.warper(im1, keypts=kp, crop=self.crop) 
                im1 = im1.to(torch.uint8)
                im2 = im2.to(torch.uint8)
                C, H, W = im1.shape
                im1 = TF.to_pil_image(im1)
                im2 = TF.to_pil_image(im2)
                im1 = self.transforms(im1)
                im2 = self.transforms(im2)
                C, H, W = im1.shape
                num_kp, dim = kp1.shape
                data = torch.cat([im1, im2], 0) # cat
                kp = torch.cat([kp1, kp2], 0)
                kp_normalized = torch.cat((kp_normalize(H, W, kp1), kp_normalize(H, W, kp2)), 0)

        else: # supervised postprocessing
            kp = self.keypoints[index].copy()

            if self.TPS_aug:
                im1 = self.initial_transforms(im)
                im1 = TF.to_tensor(im1) * 255
                im1, kp = self.warper(im1, keypts=kp, crop=self.crop) 
                im1 = im1.to(torch.uint8)
                im1 = TF.to_pil_image(im1)
                im1 = self.transforms(im1)
                data = im1
            else:
                data = self.transforms(self.initial_transforms(im))
                if self.crop != 0: # maybe useful for datasets other than celebA/MAFL
                    data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                    kp = kp - self.crop
            
            C, H, W = data.shape
            kp = torch.tensor(kp)
            kp_normalized = kp_normalize(H, W, kp)

        if self.visualize:
            # from torchvision.utils import make_grid
            from utils.visualization import norm_range
            plt.clf()
            fig = plt.figure()
            if self.pair_image:
                if not self.TPS_aug:
                    im1, im2 = torch.split(data, [3, 3], dim=0)
                    ax = fig.add_subplot(121)
                    ax.imshow(norm_range(im1).permute(1, 2, 0).cpu().numpy())
                    ax = fig.add_subplot(122)
                    ax.imshow(norm_range(im2).permute(1, 2, 0).cpu().numpy())
                    print(im1.shape, im2.shape)
                else:
                    im1, im2 = torch.split(data, [3, 3], dim=0)
                    kp1, kp2 = torch.split(kp, [num_kp, num_kp], dim=0)
                    kp1_x, kp1_y = kp1[:, 0].numpy(), kp1[:, 1].numpy()
                    kp2_x, kp2_y = kp2[:, 0].numpy(), kp2[:, 1].numpy()

                    plt.imshow(norm_range(im1).permute(1, 2, 0).cpu().numpy())
                    plt.scatter(kp1_x, kp1_y)
                    plt.savefig(os.path.join('sanity_check', vis_name + '_1.png'), bbox_inches='tight')
                    plt.close()
                    

                    fig = plt.figure()
                    plt.imshow(norm_range(im2).permute(1, 2, 0).cpu().numpy())
                    plt.scatter(kp2_x, kp2_y)
                    plt.savefig(os.path.join('sanity_check', vis_name + '_2.png'), bbox_inches='tight')
                    plt.close()
                    
            else:
                ax = fig.add_subplot(111)
                ax.imshow(norm_range(data).permute(1, 2, 0).cpu().numpy())
                kp_x = kp[:, 0].numpy()
                kp_y = kp[:, 1].numpy()
                ax.scatter(kp_x, kp_y)
                print(data.shape)
            # plt.savefig('check_dataloader.png', bbox_inches='tight')
            plt.savefig(os.path.join('sanity_check', vis_name + '.png'), bbox_inches='tight')
            print(os.path.join(self.subdir, self.filenames[index]))
            plt.close()
        return data, kp, kp_normalized, index


class CelebAPrunedAligned_MAFLVal(CelebABase):
    eye_kp_idxs = [0, 1]
    def __init__(self, root, train=True, pair_image=True, imwidth=100, crop=15, TPS_aug=False,
                 do_augmentations=True, use_keypoints=False, use_hq_ims=True,
                 visualize=False, val_split="celeba", val_size=2000, random_erasing=False,
                 **kwargs):
        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.pair_image = pair_image
        self.visualize = visualize
        self.crop = crop
        self.TPS_aug = TPS_aug # place holder 
        self.use_keypoints = use_keypoints

        if use_hq_ims:
            subdir = "img_align_celeba_hq"
        else:
            subdir = "img_align_celeba"
        self.subdir = os.path.join(root, 'Img', subdir)

        anno = pd.read_csv(
            os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
            delim_whitespace=True)
        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltrain = pd.read_csv(os.path.join(root, 'MAFL', 'training.txt'), header=None,
                                delim_whitespace=True, index_col=0)
        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        # Ensure that we are not using mafl images
        split.loc[mafltrain.index] = 3
        split.loc[mafltest.index] = 4

        assert (split[1] == 4).sum() == 1000

        if train:
            self.data = anno.loc[split[split[1] == 0].index] # CelebA train;
        elif val_split == "celeba":
            # subsample images from CelebA val, otherwise training gets slow
            self.data = anno.loc[split[split[1] == 2].index][:val_size]
        elif val_split == "mafl":
            self.data = anno.loc[split[split[1] == 4].index]

        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ;
        # leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)
        self.filenames = list(self.data.index)

        # Move head up a bit
        initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178) 
        self.keypoints[:, :, 1] -= 30
        self.keypoints *= self.imwidth / 178.

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            # the only augmentation added on the top of DVE
            transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose(
            [initial_crop, transforms.Resize(self.imwidth)])
        if random_erasing:
            self.transforms = transforms.Compose(augmentations + [normalize, transforms.RandomErasing()])
        else:
            self.transforms = transforms.Compose(augmentations + [normalize])

    #def __len__(self):
    #    return len(self.data.index)


class CelebA_MAFLVal(CelebABase):
    eye_kp_idxs = [0, 1]
    def __init__(self, root, train=True, pair_image=True, imwidth=100, crop=15, TPS_aug=False,
                 do_augmentations=True, use_keypoints=False, use_hq_ims=True,
                 visualize=False, val_split="celeba", val_size=2000, random_erasing=False,
                 **kwargs):
        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.pair_image = pair_image
        self.visualize = visualize
        self.crop = crop
        self.TPS_aug = TPS_aug # place holder 
        self.use_keypoints = use_keypoints

        print('CelebA images in the wild are used!')
    
        #if use_hq_ims:
        #    subdir = "img_align_celeba_hq"
        #else:
        #    subdir = "img_align_celeba"
        self.subdir = os.path.join(root, 'Img_in_the_wild')

        #anno = pd.read_csv(
        #    os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt'), header=1,
        #    delim_whitespace=True)

        anno = pd.read_csv(
            os.path.join(root, 'Anno', 'list_landmarks_celeba.txt'), header=1,
            delim_whitespace=True)

        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599

        mafltrain = pd.read_csv(os.path.join(root, 'MAFL', 'training.txt'), header=None,
                                delim_whitespace=True, index_col=0)
        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        # Ensure that we are not using mafl images
        split.loc[mafltrain.index] = 3
        split.loc[mafltest.index] = 4

        assert (split[1] == 4).sum() == 1000

        if train:
            self.data = anno.loc[split[split[1] == 0].index] # CelebA train;
        elif val_split == "celeba":
            # subsample images from CelebA val, otherwise training gets slow
            self.data = anno.loc[split[split[1] == 2].index][:val_size]
        elif val_split == "mafl":
            self.data = anno.loc[split[split[1] == 4].index]

        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ;
        # leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)
        self.filenames = list(self.data.index)

        # Move head up a bit
        # initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178) 
        # self.keypoints[:, :, 1] -= 30
        # self.keypoints *= self.imwidth / 178.

        # TODO the keypoints are wrongly scaled, but we don't use them
        self.keypoints[:, :, 0] *= self.imwidth / 178. 
        self.keypoints[:, :, 1] *= self.imwidth / 218  

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            # the only augmentation added on the top of DVE
            transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        #self.initial_transforms = transforms.Compose(
        #    [initial_crop, transforms.Resize(self.imwidth)])
        self.initial_transforms = transforms.Compose(
            [transforms.Resize((self.imwidth, self.imwidth))])

        if random_erasing:
            self.transforms = transforms.Compose(augmentations + [normalize, transforms.RandomErasing()])
        else:
            self.transforms = transforms.Compose(augmentations + [normalize])

    #def __len__(self):
    #    return len(self.data.index)


class CelebABaseWild(Dataset):

    def __len__(self):
        return len(self.filenames)

    def restrict_annos(self, num, datapath=None, outpath=None, repeat_flag=True, num_per_epoch=1000):

        if datapath is None:
            anno_count = len(self.filenames)
            pick = np.random.choice(anno_count, num, replace=False)
            # print(f"Picking annotation for images: {np.array(self.filenames)[pick].tolist()}")
            if repeat_flag:
                total_count = min(num_per_epoch, len(self.filenames))
                # total_count = len(self.filenames)
                assert num < total_count
                repeat = int(total_count // num)
                self.filenames = np.tile(np.array(self.filenames)[pick], repeat)
                self.keypoints = np.tile(self.keypoints[pick], (repeat, 1, 1))
            else:
                self.filenames = np.array(self.filenames)[pick]
                self.keypoints = self.keypoints[pick]

            with open(os.path.join(outpath, 'datalist.txt'), 'w') as f:
                f.writelines("%d\n" % idx for idx in pick)
            with open(os.path.join(outpath, 'namelist.txt'), 'w') as f:
                f.writelines("%s\n" % name for name in self.filenames)
        else:
            with open(os.path.join(datapath, 'datalist.txt'), 'r') as f:
                pick = np.array([int(line.rstrip()) for line in f.readlines()])
            if repeat_flag:
                total_count = min(num_per_epoch, len(self.filenames))
                # total_count = 1000 if len(self.filenames) > 1000 else len(self.filenames)
                assert num < total_count # only repeat if the number of images are too small
                repeat = int(total_count // num)
                self.filenames = np.tile(np.array(self.filenames)[pick], repeat)
                self.keypoints = np.tile(self.keypoints[pick], (repeat, 1, 1))
            else:
                self.filenames = np.array(self.filenames)[pick]
                self.keypoints = self.keypoints[pick]
            # for sanity check
            with open(os.path.join(datapath, 'namelist_resume.txt'), 'w') as f: 
                f.writelines("%s\n" % name for name in self.filenames)


    def __getitem__(self, index):
        im = Image.open(os.path.join(self.subdir, self.filenames[index])).convert("RGB")
        kp = -1
        kp_normalized = -1 # None
        # import pdb; pdb.set_trace()
        if self.pair_image: # unsupervised contrastive learning 
            if not self.TPS_aug:
                kp = self.keypoints[index].copy()
                # randomresizecrop is the key to generate pairs of images
                img1 = self.transforms(self.initial_transforms(im))
                img2 = self.transforms(self.initial_transforms(im))
                data = torch.cat([img1, img2], dim=0)
                if self.crop != 0: # maybe useful for datasets other than celebA/MAFL
                    data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                    kp = kp - self.crop
                # the following keypoints assuming there is not augmentation applied to images (random crops, resize etc.)
                kp1 = torch.tensor(kp)
                kp2 = kp1.clone()
                kp = torch.cat([kp1, kp2], 0)
                C, H, W = data.shape
                kp_normalized = torch.cat((kp_normalize(H, W, kp1), kp_normalize(H, W, kp2)), 0)
            else: 
                #  add TPS deformation for image matching, returns a pair of images, a.w.a keypoints  
                kp = self.keypoints[index].copy()
                im1 = self.initial_transforms(im)
                im1 = TF.to_tensor(im1) * 255
                im1, im2, flow, grid, kp1, kp2 = self.warper(im1, keypts=kp, crop=self.crop) 
                im1 = im1.to(torch.uint8)
                im2 = im2.to(torch.uint8)
                C, H, W = im1.shape
                im1 = TF.to_pil_image(im1)
                im2 = TF.to_pil_image(im2)
                im1 = self.transforms(im1)
                im2 = self.transforms(im2)
                C, H, W = im1.shape
                num_kp, dim = kp1.shape
                data = torch.cat([im1, im2], 0) # cat
                kp = torch.cat([kp1, kp2], 0)
                kp_normalized = torch.cat((kp_normalize(H, W, kp1), kp_normalize(H, W, kp2)), 0)

        else: # supervised postprocessing
            kp = self.keypoints[index].copy()
            imW, imH = im.size

            if self.TPS_aug:
                im1 = self.initial_transforms(im)
                im1 = TF.to_tensor(im1) * 255

                # TODO: sanity check
                # hack: resize the keypoints
                kp[:, 0] *= self.imwidth / imW
                kp[:, 1] *= self.imwidth / imH

                im1, kp = self.warper(im1, keypts=kp, crop=self.crop) 
                im1 = im1.to(torch.uint8)
                im1 = TF.to_pil_image(im1)
                im1 = self.transforms(im1)
                data = im1
            else:
                data = self.transforms(self.initial_transforms(im))

                # TODO: sanity check
                # hack: resize the keypoints
                kp[:, 0] *= self.imwidth / imW
                kp[:, 1] *= self.imwidth / imH
                
                if self.crop != 0: # maybe useful for datasets other than celebA/MAFL
                    data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                    kp = kp - self.crop
            
            C, H, W = data.shape
            kp = torch.tensor(kp)
            kp_normalized = kp_normalize(H, W, kp)

        if self.visualize:
            # from torchvision.utils import make_grid
            from utils.visualization import norm_range
            plt.clf()
            fig = plt.figure()
            if self.pair_image:
                if not self.TPS_aug:
                    im1, im2 = torch.split(data, [3, 3], dim=0)
                    ax = fig.add_subplot(121)
                    ax.imshow(norm_range(im1).permute(1, 2, 0).cpu().numpy())
                    ax = fig.add_subplot(122)
                    ax.imshow(norm_range(im2).permute(1, 2, 0).cpu().numpy())
                    print(im1.shape, im2.shape)
                else:
                    im1, im2 = torch.split(data, [3, 3], dim=0)
                    kp1, kp2 = torch.split(kp, [num_kp, num_kp], dim=0)
                    kp1_x, kp1_y = kp1[:, 0].numpy(), kp1[:, 1].numpy()
                    kp2_x, kp2_y = kp2[:, 0].numpy(), kp2[:, 1].numpy()

                    plt.imshow(norm_range(im1).permute(1, 2, 0).cpu().numpy())
                    plt.scatter(kp1_x, kp1_y)
                    plt.savefig(os.path.join('sanity_check', vis_name + '_1.png'), bbox_inches='tight')
                    plt.close()
                    

                    fig = plt.figure()
                    plt.imshow(norm_range(im2).permute(1, 2, 0).cpu().numpy())
                    plt.scatter(kp2_x, kp2_y)
                    plt.savefig(os.path.join('sanity_check', vis_name + '_2.png'), bbox_inches='tight')
                    plt.close()
                    
            else:
                ax = fig.add_subplot(111)
                ax.imshow(norm_range(data).permute(1, 2, 0).cpu().numpy())
                kp_x = kp[:, 0].numpy()
                kp_y = kp[:, 1].numpy()
                ax.scatter(kp_x, kp_y)
                print(data.shape)
            # plt.savefig('check_dataloader.png', bbox_inches='tight')
            plt.savefig(os.path.join('sanity_check', vis_name + '.png'), bbox_inches='tight')
            print(os.path.join(self.subdir, self.filenames[index]))
            plt.close()
        return data, kp, kp_normalized, index


class MAFL_wild(CelebABaseWild):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_image=False, imwidth=100, crop=15,
                 do_augmentations=False, use_hq_ims=True, visualize=False, TPS_aug=False, **kwargs):
        self.root = root
        self.imwidth = imwidth
        self.use_hq_ims = use_hq_ims
        self.visualize = visualize
        self.train = train
        self.pair_image = pair_image
        self.crop = crop
        # subdir = "img_align_celeba_hq" if use_hq_ims else "img_align_celeba"
        # self.subdir = os.path.join(root, 'Img', subdir)
        self.subdir = os.path.join(root, 'Img_in_the_wild')
        # annos_path = os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt')
        annos_path = os.path.join(root, 'Anno', 'list_landmarks_celeba.txt')
        anno = pd.read_csv(annos_path , header=1, delim_whitespace=True)
        self.TPS_aug = TPS_aug
        if self.pair_image:
            warp_kwargs = dict(
                warpsd_all=0.001 * .5,
                warpsd_subset=0.01 * .5,
                transsd=0.1 * .5,
                scalesd=0.1 * .5,
                rotsd=5 * .5,
                im1_multiplier=1,
                im1_multiplier_aff=1
            )
            self.warper = tps.Warper(imwidth, imwidth, **warp_kwargs) # used for image matching experiments
        else:
            self.warper = tps.WarperSingle(H=imwidth, W=imwidth)

        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599
        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        split.loc[mafltest.index] = 4
        mafltrain = pd.read_csv(os.path.join(root, 'MAFL', 'training.txt'), header=None,
                                delim_whitespace=True, index_col=0)
        split.loc[mafltrain.index] = 5
        assert (split[1] == 4).sum() == 1000
        assert (split[1] == 5).sum() == 19000

        if train:
            self.data = anno.loc[split[split[1] == 5].index]
        else:
            self.data = anno.loc[split[split[1] == 4].index]

        # keypoint ordering
        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ;
        # leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)
        self.filenames = list(self.data.index)

        # comment out the following codes to disable initial cropping

        # Move head up a bit
        # initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178)
        # self.keypoints[:, :, 1] -= 30
        # self.keypoints *= self.imwidth / 178.


        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose(
            [transforms.Resize((self.imwidth, self.imwidth))])
        self.transforms = transforms.Compose(augmentations + [normalize])



class MAFLAligned(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_image=False, imwidth=100, crop=15,
                 do_augmentations=False, use_hq_ims=True, visualize=False, TPS_aug=False, **kwargs):
        self.root = root
        self.imwidth = imwidth
        self.use_hq_ims = use_hq_ims
        self.visualize = visualize
        self.train = train
        self.pair_image = pair_image
        self.crop = crop
        subdir = "img_align_celeba_hq" if use_hq_ims else "img_align_celeba"
        self.subdir = os.path.join(root, 'Img', subdir)
        annos_path = os.path.join(root, 'Anno', 'list_landmarks_align_celeba.txt')
        anno = pd.read_csv(annos_path , header=1, delim_whitespace=True)
        self.TPS_aug = TPS_aug
        if self.pair_image:
            warp_kwargs = dict(
                warpsd_all=0.001 * .5,
                warpsd_subset=0.01 * .5,
                transsd=0.1 * .5,
                scalesd=0.1 * .5,
                rotsd=5 * .5,
                im1_multiplier=1,
                im1_multiplier_aff=1
            )
            self.warper = tps.Warper(imwidth, imwidth, **warp_kwargs) # used for image matching experiments
        else:
            self.warper = tps.WarperSingle(H=imwidth, W=imwidth)

        assert len(anno.index) == 202599
        split = pd.read_csv(os.path.join(root, 'Eval', 'list_eval_partition.txt'),
                            header=None, delim_whitespace=True, index_col=0)
        assert len(split.index) == 202599
        mafltest = pd.read_csv(os.path.join(root, 'MAFL', 'testing.txt'), header=None,
                               delim_whitespace=True, index_col=0)
        split.loc[mafltest.index] = 4
        mafltrain = pd.read_csv(os.path.join(root, 'MAFL', 'training.txt'), header=None,
                                delim_whitespace=True, index_col=0)
        split.loc[mafltrain.index] = 5
        assert (split[1] == 4).sum() == 1000
        assert (split[1] == 5).sum() == 19000

        if train:
            self.data = anno.loc[split[split[1] == 5].index]
        else:
            self.data = anno.loc[split[split[1] == 4].index]

        # keypoint ordering
        # lefteye_x lefteye_y ; righteye_x righteye_y ; nose_x nose_y ;
        # leftmouth_x leftmouth_y ; rightmouth_x rightmouth_y
        self.keypoints = np.array(self.data, dtype=np.float32).reshape(-1, 5, 2)
        self.filenames = list(self.data.index)

        # Move head up a bit
        initial_crop = lambda im: transforms.functional.crop(im, 30, 0, 178, 178)
        self.keypoints[:, :, 1] -= 30
        self.keypoints *= self.imwidth / 178.
        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose(
            [initial_crop, transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])



class AFLW(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_image=False, imwidth=100, crop=15,
                 do_augmentations=False, visualize=False, use_minival=False, TPS_aug=False, **kwargs):
        self.root = root
        self.crop = crop
        self.imwidth = imwidth
        self.visualize = visualize
        self.use_minival = use_minival
        self.train = train
        self.pair_image = pair_image
        self.TPS_aug = TPS_aug
        self.warper = tps.WarperSingle(H=imwidth, W=imwidth)

        images, keypoints, sizes = self.load_dataset(root)
        self.sizes = sizes
        self.filenames = images
        self.keypoints = keypoints.astype(np.float32)
        self.subdir = os.path.join(root, 'output')

        # print("LIMITING DATA FOR DEBGGING")
        # self.filenames = self.filenames[:1000]
        # self.keypoints = self.keypoints[:1000]
        # sizes = sizes[:1000]
        # self.sizes = sizes

        # check raw
        if False:
            im_path = pjoin(self.subdir, self.filenames[0])
            im = Image.open(im_path).convert("RGB")
            plt.imshow(im)
            plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1])
        
        self.keypoints *= self.imwidth / sizes[:, [1, 0]].reshape(-1, 1, 2)
        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        # NOTE: we break the aspect ratio here, but hopefully the network should
        # be fairly tolerant to this
        self.initial_transforms = transforms.Resize((self.imwidth, self.imwidth))
        augmentations = [
            JPEGNoise(),
            transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]
        self.transforms = transforms.Compose(augmentations + [normalize])

    def load_dataset(self, data_dir):
        # borrowed from Tom and Ankush
        if self.train or self.use_minival:
            load_subset = "train"
        else:
            load_subset = "test"
        with open(pjoin(data_dir, 'aflw_{}_images.txt'.format(load_subset)), 'r') as f:
            images = f.read().splitlines()
        mat = loadmat(os.path.join(data_dir, 'aflw_' + load_subset + '_keypoints.mat'))
        keypoints = mat['gt'][:, :, [1, 0]]
        sizes = mat['hw']

        # import ipdb; ipdb.set_trace()
        # if self.data.shape[0] == 19000:
        #     self.data = self.data[:20]

        if load_subset == 'train':
            # put the last 10 percent of the training aside for validation
            if self.use_minival:
                n_validation = int(round(0.1 * len(images)))
                if self.train:
                    images = images[:-n_validation]
                    keypoints = keypoints[:-n_validation]
                    sizes = sizes[:-n_validation]
                else:
                    images = images[-n_validation:]
                    keypoints = keypoints[-n_validation:]
                    sizes = sizes[-n_validation:]
        return images, keypoints, sizes

# AFLW rotated images
class AFLW_rotated(CelebABase):
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_image=False, imwidth=100, crop=15,
                 do_augmentations=False, visualize=False, use_minival=False, TPS_aug=False, **kwargs):
        self.root = root
        self.crop = crop
        self.imwidth = imwidth
        self.visualize = visualize
        self.use_minival = use_minival
        self.train = train
        self.pair_image = pair_image
        self.TPS_aug = TPS_aug
        self.warper = tps.WarperSingle(H=imwidth, W=imwidth)

        images, keypoints, sizes = self.load_dataset(root)
        self.sizes = sizes
        self.filenames = images
        self.keypoints = keypoints.astype(np.float32)
        self.subdir = root

        # check raw
        if False:
            im_path = pjoin(self.subdir, self.filenames[0])
            im = Image.open(im_path).convert("RGB")
            plt.imshow(im)
            plt.scatter(keypoints[0, :, 0], keypoints[0, :, 1])
        
        self.keypoints *= self.imwidth / sizes[:, [1, 0]].reshape(-1, 1, 2)
        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        # NOTE: we break the aspect ratio here, but hopefully the network should
        # be fairly tolerant to this
        self.initial_transforms = transforms.Resize((self.imwidth, self.imwidth))
        augmentations = [
            JPEGNoise(),
            transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]
        self.transforms = transforms.Compose(augmentations + [normalize])

    def load_dataset(self, data_dir):
        with open(pjoin(data_dir, 'samples.txt'), 'r') as f:
            images = f.read().splitlines()
        keypoints = np.ones((len(images), 5, 2)) * 30 # pseudo keypoint annotations
        sizes = np.ones((len(images), 2)) # pseudo sizes
        return images, keypoints, sizes


class AFLW_MTFL(CelebABase):
    """Used for testing on the 5-point version of AFLW included in the MTFL download from the
       Facial Landmark Detection by Deep Multi-task Learning (TCDCN) paper
       http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html

       For training this uses a cropped 5-point version of AFLW used in
       http://openaccess.thecvf.com/content_ICCV_2017/papers/Thewlis_Unsupervised_Learning_of_ICCV_2017_paper.pdf
       """
    eye_kp_idxs = [0, 1]

    def __init__(self, root, train=True, pair_image=False, imwidth=100, crop=15,
                 do_augmentations=False, visualize=False, TPS_aug=False, **kwargs):

        # MTFL from http://mmlab.ie.cuhk.edu.hk/projects/TCDCN/data/MTFL.zip
        self.test_root = os.path.join(root, 'MTFL')  
        # AFLW cropped from www.robots.ox.ac.uk/~jdt/aflw_10122train_cropped.zip
        self.train_root = os.path.join(root, 'aflw_cropped')  

        self.imwidth = imwidth
        self.train = train
        self.pair_image = pair_image
        self.crop = crop
        self.visualize = visualize
        initial_crop = lambda im: im
        self.TPS_aug = TPS_aug
        self.warper = tps.WarperSingle(H=imwidth, W=imwidth) 

        test_anno = pd.read_csv(os.path.join(self.test_root, 'testing.txt'),
                                header=None, delim_whitespace=True)

        if train:
            self.root = self.train_root
            all_anno = pd.read_csv(os.path.join(self.train_root, 'facedata_cropped.csv'),
                                   sep=',', header=0)
            allims = all_anno.image_file.to_list()
            trainims = all_anno[all_anno.set == 1].image_file.to_list()
            testims = [t.split('-')[-1] for t in test_anno.loc[:, 0].to_list()]

            for x in trainims:
                assert x not in testims

            for x in testims:
                assert x in allims

            self.filenames = all_anno[all_anno.set == 1].crop_file.to_list()
            self.keypoints = np.array(all_anno[all_anno.set == 1].iloc[:, 4:14],
                                      dtype=np.float32).reshape(-1, 5, 2)

            self.keypoints -= 1  # matlab to python
            self.keypoints *= self.imwidth / 150.

            assert len(self.filenames) == 10122
        else:
            self.root = self.test_root
            keypoints = np.array(test_anno.iloc[:, 1:11], dtype=np.float32)
            self.keypoints = keypoints.reshape(-1, 2, 5).transpose(0, 2, 1)
            self.filenames = test_anno[0].to_list()

            self.keypoints -= 1  # matlab to python
            self.keypoints *= self.imwidth / 150.

            assert len(self.filenames) == 2995
        self.subdir = self.root

        # print("HARDCODING DEBGGER")
        # self.filenames = self.filenames[:100]
        # self.keypoints = self.keypoints[:100]

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769],
                                         std=[0.2599, 0.2371, 0.2323])
        augmentations = [
            JPEGNoise(),
            # transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
            transforms.transforms.ColorJitter(.4, .4, .4),
            transforms.ToTensor(),
            PcaAug()
        ] if (train and do_augmentations) else [transforms.ToTensor()]
        self.initial_transforms = transforms.Compose(
            [initial_crop, transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])



class ThreeHundredW(Dataset):
    """The 300W dataset, which is an amalgamation of several other datasets

    We use the split from "Face alignment at 3000 fps via regressing local binary features"
    Where they state:
    "Our training set consists of AFW, the training sets of LFPW,
    and the training sets of Helen,  with 3148 images in total.
    Our testing set consists of IBUG, the testing sets of LFPW,
    and the testing sets of Helen, with 689 images in total.
    We do not use images from XM2VTS as it is taken under a
    controlled environment and is too simple"
    """
    eye_kp_idxs = [36, 45]

    def __init__(self, root, train=True, pair_image=None, imwidth=100, crop=20,
                 do_augmentations=False, use_keypoints=False, visualize=False, **kwargs):

        from scipy.io import loadmat

        self.root = root
        self.imwidth = imwidth
        self.train = train
        self.pair_image = pair_image
        self.crop = crop
        self.visualize = visualize

        afw = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_afw.mat'))
        helentr = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_helen_trainset.mat'))
        helente = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_helen_testset.mat'))
        lfpwtr = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_lfpw_trainset.mat'))
        lfpwte = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_lfpw_testset.mat'))
        ibug = loadmat(os.path.join(root, 'Bounding Boxes/bounding_boxes_ibug.mat'))

        self.filenames = []
        self.bounding_boxes = []
        self.keypoints = []

        if train:
            datasets = [(afw, 'afw'), (helentr, 'helen/trainset'), (lfpwtr, 'lfpw/trainset')]
        else:
            datasets = [(helente, 'helen/testset'), (lfpwte, 'lfpw/testset'), (ibug, 'ibug')]

        for dset in datasets:
            ds = dset[0]
            ds_imroot = dset[1]
            imnames = [ds['bounding_boxes'][0, i]['imgName'][0, 0][0] for i in range(ds['bounding_boxes'].shape[1])]
            bbs = [ds['bounding_boxes'][0, i]['bb_ground_truth'][0, 0][0] for i in range(ds['bounding_boxes'].shape[1])]

            for i, imn in enumerate(imnames):
                # only some of the images given in ibug boxes exist (those that start with 'image')
                if ds is not ibug or imn.startswith('image'):
                    self.filenames.append(os.path.join(ds_imroot, imn))
                    self.bounding_boxes.append(bbs[i])

                    kpfile = os.path.join(root, ds_imroot, imn[:-3] + 'pts')
                    with open(kpfile) as kpf:
                        kp = kpf.read()
                    kp = kp.split()[5:-1]
                    kp = [float(k) for k in kp]
                    assert len(kp) == 68 * 2
                    kp = np.array(kp).astype(np.float32).reshape(-1, 2)
                    self.keypoints.append(kp)

        if train:
            assert len(self.filenames) == 3148
        else:
            assert len(self.filenames) == 689

        normalize = transforms.Normalize(mean=[0.5084, 0.4224, 0.3769], std=[0.2599, 0.2371, 0.2323])
        augmentations = [JPEGNoise(), 
                         transforms.RandomResizedCrop(self.imwidth, scale=(0.2, 1.)),  
                         transforms.transforms.ColorJitter(.4, .4, .4),
                         transforms.ToTensor(), 
                         PcaAug()] if (train and do_augmentations) else [transforms.ToTensor()]

        self.initial_transforms = transforms.Compose([transforms.Resize(self.imwidth)])
        self.transforms = transforms.Compose(augmentations + [normalize])

        # print("HARDCODING DEBGGER")
        # self.filenames = self.filenames[:100]
        # self.keypoints = self.keypoints[:100]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        im = Image.open(os.path.join(self.root, self.filenames[index])).convert("RGB")
        kp_normalized = -1 # None
        
        # Crop bounding box
        xmin, ymin, xmax, ymax = self.bounding_boxes[index]
        keypts = self.keypoints[index]

        # This is basically copied from matlab code and assumes matlab indexing
        bw = xmax - xmin + 1
        bh = ymax - ymin + 1
        bcy = ymin + (bh + 1) / 2
        bcx = xmin + (bw + 1) / 2

        # To simplify the preprocessing, we do two image resizes (can fix later if speed
        # is an issue)
        preresize_sz = 100

        bw_ = 52  # make the (tightly cropped) face 52px
        fac = bw_ / bw
        imr = im.resize((int(im.width * fac), int(im.height * fac)))

        bcx_ = int(np.floor(fac * bcx))
        bcy_ = int(np.floor(fac * bcy))
        bx = bcx_ - bw_ / 2 + 1
        bX = bcx_ + bw_ / 2
        by = bcy_ - bw_ / 2 + 1
        bY = bcy_ + bw_ / 2
        pp = (preresize_sz - bw_) / 2
        bx = int(bx - pp)
        bX = int(bX + pp)
        by = int(by - pp - 2)
        bY = int(bY + pp - 2)

        imr = pad_and_crop(np.array(imr), [(by - 1), bY, (bx - 1), bX])
        im = Image.fromarray(imr)

        cutl = bx - 1
        keypts = keypts.copy() * fac
        keypts[:, 0] = keypts[:, 0] - cutl
        cutt = by - 1
        keypts[:, 1] = keypts[:, 1] - cutt

        kp = keypts - 1  # from matlab to python style
        kp = kp * self.imwidth / preresize_sz
        kp = torch.tensor(kp)

        if self.pair_image: # unsupervised contrastive learning 
            # randomresizecrop is the key to generate pairs of images
            img1 = self.transforms(self.initial_transforms(im))
            img2 = self.transforms(self.initial_transforms(im))
            data = torch.cat([img1, img2], dim=0)
            if self.crop != 0: # maybe useful for datasets other than celebA/MAFL
                data = data[:, self.crop:-self.crop, self.crop:-self.crop]
        else: # supervised postprocessing
            data = self.transforms(self.initial_transforms(im))
            if self.crop != 0: # maybe useful for datasets other than celebA/MAFL
                data = data[:, self.crop:-self.crop, self.crop:-self.crop]
                kp = kp - self.crop
            C, H, W = data.shape
            # kp = torch.tensor(kp)
            kp = torch.as_tensor(kp)
            kp_normalized = kp_normalize(H, W, kp)
           

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
            else:
                ax = fig.add_subplot(111)
                ax.imshow(norm_range(data).permute(1, 2, 0).cpu().numpy())
                kp_x = kp[:, 0].numpy()
                kp_y = kp[:, 1].numpy()
                ax.scatter(kp_x, kp_y)
            plt.savefig('check_dataloader.png')
            print(os.path.join(self.root, self.filenames[index]))
            plt.close()

        return data, kp, kp_normalized, index



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Helen")
    parser.add_argument("--use_keypoints", action="store_true")
    parser.add_argument("--use_ims", type=int, default=1)
    parser.add_argument("--use_minival", action="store_true")
    parser.add_argument("--break_preproc", action="store_true")
    parser.add_argument("--pairs", action="store_true")
    parser.add_argument("--rand_in", action="store_true")
    parser.add_argument("--restrict_to", type=int, help="restrict to n images")
    parser.add_argument("--downsample_labels", type=int, default=2)
    parser.add_argument("--show", type=int, default=2)
    parser.add_argument("--restrict_seed", type=int, default=0)
    parser.add_argument("--root")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--pair_image", action="store_true")
    parser.add_argument("--TPS_aug", action="store_true")
    parser.add_argument("--do_augmentations", action="store_true")
    parser.add_argument("--vis_name", type=str, default='check_dataloader.png')
    args = parser.parse_args()

    default_roots = {
        "CelebAPrunedAligned_MAFLVal": "./datasets/celeba",
        "CelebA_MAFLVal": "./datasets/celeba",
        "MAFLAligned": "./datasets/celeba",
        "MAFL_wild": "./datasets/celeba",
        "AFLW_MTFL": "./datasets/face_datasets/aflw-mtfl",
        "AFLW": "./datasets/face_datasets/aflw/aflw_release-2",
        "ThreeHundredW": "./datasets/face_datasets/300w/300w",
    }
    root = default_roots[args.dataset] if args.root is None else args.root

    imwidth = 136
    crop = 20
    if args.dataset == 'MAFL_wild':
        crop = 20
    # The following three flags have to be true at the same time during contrastive learning  
    #train = True # the split to use
    #pair_image = True # unsupervised learning 
    #do_augmentations = True
    vis_name = args.vis_name
    kwargs = {
        "root": root,
        "train": args.train,
        "pair_image": args.pair_image,
        'do_augmentations': args.do_augmentations,
        'TPS_aug' : args.TPS_aug,
        'vis_name': args.vis_name,
        "use_keypoints": args.use_keypoints,
        "use_ims": args.use_ims,
        "visualize": True,
        "use_minival": args.use_minival,
        "downsample_labels": args.downsample_labels,
        "break_preproc": args.break_preproc,
        "rand_in": args.rand_in,
        "restrict_to": args.restrict_to,
        "restrict_seed": args.restrict_seed,
        "imwidth": imwidth,
        "crop": crop,
    }
    if args.train and args.pairs:
        kwargs["pair_warper"] = True
    
    torch.manual_seed(0)
    show = args.show
    if args.restrict_to:
        show = min(args.restrict_to, show)
    if args.dataset == "IJBB":
        dataset = IJBB('data/ijbb', prototypes=True, imwidth=128, train=False)
        for ii in range(show):
            dataset[ii]
    else:
        dataset = globals()[args.dataset](**kwargs)
        #for ii in range(show):
        dataset[3]
