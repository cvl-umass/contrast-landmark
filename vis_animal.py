"""
evaluate pretrained MoCo and landmark regressor
on animal benchmarks (e.g. CUB)
"""

from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import argparse
import socket
import torch.multiprocessing as mp
import torch.distributed as dist

import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from utils.util import adjust_learning_rate, AverageMeter, Tee
from utils import clean_state_dict, Logger

from models.resnet import InsResNet50,InsResNet18,InsResNet34,InsResNet101,InsResNet152
from models.hourglass import HourglassNet
from models.keypoint_prediction import IntermediateKeypointPredictor
from models.loss import regression_loss, selected_regression_loss
from models.metric import calc_pck
import data_loader.data_loaders_animal as module_data
import numpy as np
from utils.visualization import keypoints_animal,keypoints_animal_HR, plot_gallery

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')
    parser.add_argument('--vis_path', type=str, help='the path to store the visualization results')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet50x2', 'resnet50x4', 'hourglass',
                                 'resnet18', 'resnet34', 'resnet101', 'resnet152'])

    parser.add_argument('--trained_model_path', type=str, default=None, help='the model to test')
    parser.add_argument('--layer', type=int, default=3, help='which layer to evaluate')

    # crop
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop')
    parser.add_argument('--image_crop', type=int, default=15, help='image pre-crop') # image preprocessing
    parser.add_argument('--image_size', type=int, default=100, help='image size') # image preprocessing

    # dataset
    parser.add_argument('--dataset', type=str, default='CUB')

    # model path and name  
    parser.add_argument('--ckpt_path', type=str, help='pretrained landmark regressor') # pretrained ckpt

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # add BN
    parser.add_argument('--bn', action='store_true', help='use parameter-free BN')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')
    parser.add_argument('--multistep', action='store_true', help='use multistep LR')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--amsgrad', action='store_true', help='use amsgrad for adam')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    # log_path
    parser.add_argument('--log_path', default='log_tmp', type=str, metavar='PATH', help='path to the log file')

    # use hypercolumn or single layer output
    parser.add_argument('--use_hypercol', action='store_true', help='use hypercolumn as representations')

    # visualization
    parser.add_argument('--vis_keypoints', action='store_true', help='visualize the keypoint predictions')
    parser.add_argument('--vis_PCA', action='store_true', help='visualize the PCA projections of representations')

    opt = parser.parse_args()


    num_annotated_points = {
        "CUB": 15
    }
    opt.data_folder = './datasets/CUB_200_2011'
    opt.num_points = num_annotated_points[opt.dataset]
    Tee(opt.log_path, 'a')

    return opt


def main():

    args = parse_option()

    test_dataset   = getattr(module_data, args.dataset)(
                                         args.data_folder, 
                                         test=True, 
                                         pair_image=False, 
                                         imwidth=args.image_size, 
                                         crop=args.image_crop)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    print('Number of test images: %d' % len(test_dataset))

    input_size = args.image_size - 2 * args.image_crop
    pool_size = int(input_size / 2**5) # 96x96 --> 3; 160x160 --> 5; 224x224 --> 7;
    args.output_shape = (48,48)
    args.boxsize = 48

    if args.model == 'resnet50':
        model = InsResNet50(pool_size=pool_size)
        desc_dim = {1:64, 2:256, 3:512, 4:1024, 5:2048}
    elif args.model == 'resnet50x2':
        model = InsResNet50(width=2, pool_size=pool_size)
        desc_dim = {1:128, 2:512, 3:1024, 4:2048, 5:4096}
    elif args.model == 'resnet50x4':
        model = InsResNet50(width=4, pool_size=pool_size)
        desc_dim = {1:512, 2:1024, 3:2048, 4:4096, 5:8192}
    elif args.model == 'resnet18':
        model = InsResNet18(width=1, pool_size=pool_size)
        desc_dim = {1:64, 2:64, 3:128, 4:256, 5:512}
    elif args.model == 'resnet34':
        model = InsResNet34(width=1, pool_size=pool_size)
        desc_dim = {1:64, 2:64, 3:128, 4:256, 5:512}
    elif args.model == 'resnet101':
        model = InsResNet101(width=1, pool_size=pool_size)
        desc_dim = {1:64, 2:256, 3:512, 4:1024, 5:2048}
    elif args.model == 'resnet152':
        model = InsResNet152(width=1, pool_size=pool_size)
        desc_dim = {1:64, 2:256, 3:512, 4:1024, 5:2048}
    elif args.model == 'hourglass':
        model = HourglassNet()
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    if args.model == 'hourglass':
        feat_dim = 64
    else:
        if args.use_hypercol:
            feat_dim = 0
            for i in range(args.layer):
                feat_dim += desc_dim[5-i]
        else:
            feat_dim = desc_dim[args.layer]
    args.feat_dim = feat_dim

    print('==> loading pre-trained MOCO')
    ckpt = torch.load(args.trained_model_path, map_location='cpu')
    if args.model == 'hourglass':
        model.load_state_dict(clean_state_dict(ckpt["state_dict"])) 
    else:
        model.load_state_dict(ckpt['model'], strict=False)
    print("==> loaded checkpoint '{}'".format(args.trained_model_path))
    print('==> done')


    model = model.cuda()
    cudnn.benchmark = True

    if args.vis_PCA:
        PCA(test_loader, model, args)
    else:
        criterion = selected_regression_loss
        regressor =  IntermediateKeypointPredictor(feat_dim, num_annotated_points=args.num_points, 
                                                    num_intermediate_points=50, 
                                                    softargmax_mul = 100.0)
        regressor = regressor.cuda()
        print('==> loading pre-trained landmark regressor {}'.format(args.ckpt_path))
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        regressor.load_state_dict(checkpoint['regressor'])
        del checkpoint
        torch.cuda.empty_cache()
        test_PCK, test_loss = validate(test_loader, model, regressor, criterion, args)


def validate(val_loader, model, regressor, criterion, opt):
    batch_time = AverageMeter()
    losses = AverageMeter()
    PCK = AverageMeter()

    # switch to evaluate mode
    model.eval()
    regressor.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, visible, target, index) in enumerate(val_loader):
            input = input.cuda(opt.gpu, non_blocking=True)
            input = input.float()
            target = target.cuda(opt.gpu, non_blocking=True)
            target = target.float()

            # compute output
            if opt.model == 'hourglass':
                feat = model(input, opt.layer, opt.output_shape)
            else:
                feat = model(input, opt.layer, opt.use_hypercol, opt.output_shape)
            feat = feat.detach()
            output, _ = regressor(feat)
            loss = criterion(output, target, visible)

            # measure accuracy and record loss
            ic_error = calc_pck(output, target, visible, boxsize=opt.boxsize)
            losses.update(loss.item(), input.size(0))
            PCK.update(ic_error, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if opt.vis_keypoints and idx <= 3:
                keypoints_animal(input, output, target, visible, index, opt.vis_path) 

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'PCK {PCK.val:.3f}'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    PCK=PCK))

        print(' * PCK {PCK.avg:.3f}'.format(PCK=PCK))

    return PCK.avg, losses.avg



def PCA(val_loader, model, args):

    feat = [] 
    with torch.no_grad():
        for idx, (input, _, target, index) in enumerate(val_loader):
            input = input.float()
            input = input.cuda(args.gpu, non_blocking=True)
            # compute output
            # feat_batch = model(input, args.layer, args.use_hypercol, args.output_shape)
            if args.model == 'hourglass':
                feat_batch = model(input, args.layer, args.output_shape)
            else:
                feat_batch = model(input, args.layer, args.use_hypercol, args.output_shape)
            feat_batch = feat_batch.detach().cpu().numpy()

            feat.append(feat_batch)
            # only use one batch of images 
            if idx == 0:
                break
    feat = np.concatenate(np.array(feat), axis=0)

    print('Shape of feat: {}'.format(feat.shape))

    # feature map: 32x3840x48x48, reshape 3840x(32x48x48)
    feat = np.transpose(feat, (1, 0, 2, 3))
    assert args.feat_dim == feat.shape[0]
    feat = feat.reshape((args.feat_dim, -1)).T # n_samples x 3840

    # PCA decomposition
    from sklearn.decomposition import PCA
    t0 = time.time()
    n_components = 25
    pca = PCA(n_components=n_components).fit(feat)
    print("done in %0.3fs" % (time.time() - t0))
    feat_pca = pca.transform(feat) # n_samples x n_comp
    
    # reshape back to 32xn_compx48x48  
    feat_pca = feat_pca.T.reshape((n_components, 32,48,48))
    feat_pca = np.transpose(feat_pca, (1, 0, 2, 3))

    # visualization
    N,C,H,W = feat_pca.shape
    titles = ['basis %d' % i for i in range(n_components)]
    for idx in range(N):
        # outpath = os.path.join(args.vis_path, '%d.png' % idx)
        # plot_gallery(feat_pca[idx], titles, outpath)
        outpath = os.path.join(args.vis_path, str(idx))
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plot_gallery(feat_pca[idx], input[idx],  outpath)


if __name__ == '__main__':
    main()
