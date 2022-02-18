from __future__ import print_function
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import tensorboard_logger as tb_logger
from torchvision import transforms, datasets
from utils.util import adjust_learning_rate, AverageMeter, Tee
from utils import tps
from models.resnet import InsResNet50,InsResNet18,InsResNet34,InsResNet101,InsResNet152
from models.hourglass import HourglassNet
from models.feat_distiller import FeatDistiller
from data_loader.data_loaders_face import CelebAPrunedAligned_MAFLVal, MAFLAligned
import matplotlib.pyplot as plt
from utils.visualization import norm_range
import numpy as np

def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, 
                        default='30,40,50', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, 
                        default=0.2, help='decay rate for learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet50_half', 'resnet50x2', 'resnet50x4', 
                        'hourglass','resnet18', 'resnet34', 'resnet101', 'resnet152'])
    parser.add_argument('--trained_model_path', type=str, default=None, help='pretrained moco')
    parser.add_argument('--train_layer', type=int, default=4, help='num layer in hypercol')
    parser.add_argument('--val_layer', type=int, default=4, help='num layer in hypercol')

    # crop
    parser.add_argument('--image_crop', type=int, default=20, help='image pre-crop') 
    parser.add_argument('--image_size', type=int, default=136, help='image size') 
    parser.add_argument('--train_out_size', type=int, default=24, help='output size') 
    parser.add_argument('--val_out_size', type=int, default=96, help='output size')

    # dataset
    # fine-tune the pretrained moco on CelebA dataset 
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['MAFLAligned', 'AFLW_MTFL', 'AFLW', 'ThreeHundredW', 'InatAve', 'CelebA'])
    parser.add_argument('--val_dataset', type=str, default='MAFLAligned', choices=['MAFLAligned', 'AFLW_MTFL', 'AFLW', 'ThreeHundredW', 'InatAve', 'CelebA'], help='dataset used for image matching experiments')

    # model path and name  
    parser.add_argument('--model_name', type=str, default='feature projector') 
    parser.add_argument('--model_path', type=str, default='./logs') # path to store the models

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # augmentation
    parser.add_argument('--TPS_aug', action='store_true', help='Thin-Plate Spline augmentation')

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
    parser.add_argument('--vis_path', type=str, metavar='PATH', help='path to save visualization results')

    # use hypercolumn or single layer output
    parser.add_argument('--train_use_hypercol', action='store_true', help='use HC as representations during training')
    parser.add_argument('--val_use_hypercol', action='store_true', help='use HC as representations during testing')

    # feature distillation
    parser.add_argument('--feat_distill', action='store_true', help='feature distillation')
    parser.add_argument('--distill_mode', type=str, default='softmax', 
                        choices=['softplus', 'softmax', 'linear'], help='mode of heatmap')
    parser.add_argument('--kernel_size', type=int, default=1, help='kernel_size in the feature distiller')
    parser.add_argument('--out_dim', type=int, default=64, help='dim of feature distiller output')
    parser.add_argument('--softargmax_mul', type=float, default=7., help='temparture hyperparameters in feature distiller')
    parser.add_argument('--temperature', type=float, default=7., help='temparture hyperparameters for dense corr loss')
    parser.add_argument('--trained_feat_model_path', type=str, default=None, help='the pretrained feat model to test')
    parser.add_argument('--evaluation_mode', action='store_true', help='evaluate pretrained feature distiller')
    parser.add_argument('--visualize_matching', action='store_true', help='evaluate pretrained feature distiller')

    opt = parser.parse_args()

    # set the path according to the environment
    default_roots = {
        "MAFLAligned": "./datasets/celeba",
        "AFLW_MTFL": "./datasets/face_datasets/aflw-mtfl",
        "AFLW": "./datasets/face_datasets/aflw/aflw_release-2",
        "ThreeHundredW": "./datasets/face_datasets/300w/300w",
    }
    eye_idxs = {
        "MAFLAligned": [0, 1],
        "AFLW_MTFL": [0, 1],
        "AFLW": [0, 1],
        "ThreeHundredW": [36, 45]
    }
    num_annotated_points = {
        "MAFLAligned": 5,
        "AFLW_MTFL": 5,
        "AFLW": 5,
        "ThreeHundredW": 68
    }

    opt.data_folder = default_roots[opt.val_dataset]
    opt.eye_idx = eye_idxs[opt.val_dataset]
    opt.num_points = num_annotated_points[opt.val_dataset]
    Tee(opt.log_path, 'a')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.save_path = opt.model_path
    opt.tb_path = '%s_tensorboard' % opt.model_path
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)
    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():

    global best_error
    best_error = np.Inf

    args = parse_option()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    torch.manual_seed(0)

    # train on celebA unlabeled dataset
    train_dataset = CelebAPrunedAligned_MAFLVal(args.data_folder, 
                                                train=True, 
                                                pair_image=False, 
                                                do_augmentations=True,
                                                imwidth=args.image_size,
                                                crop = args.image_crop)
    print('Number of training images: %d' % len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, sampler=None)

    
    # validation set from MAFLAligned trainset for hyperparameter searching
    # we sample 2000 images as our val set
    val_dataset   = MAFLAligned(args.data_folder, 
                                train=True, # train set
                                pair_image=True, 
                                do_augmentations=False,
                                TPS_aug = True, 
                                imwidth=args.image_size, 
                                crop=args.image_crop)
    print('Initial number of validation images: %d' % len(val_dataset)) 
    val_dataset.restrict_annos(num=2000, outpath=args.save_folder, repeat_flag=False)
    print('After restricting the size of validation set: %d' % len(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)


    # testing set from MAFLAligned test for evaluating image matching
    test_dataset = MAFLAligned(args.data_folder, 
                               train=False, # test set 
                               pair_image=True, 
                               do_augmentations=False,
                               TPS_aug = True, # match landmark between deformed images
                               imwidth=args.image_size, 
                               crop=args.image_crop)
    print('Number of testing images: %d' % len(test_dataset)) 
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    assert len(val_dataset) == 2000
    assert len(test_dataset) == 1000


    # create model and optimizer
    input_size = args.image_size - 2 * args.image_crop
    pool_size = int(input_size / 2**5) # 96x96 --> 3; 160x160 --> 5; 224x224 --> 7;
    # we use smaller feature map when training the feature distiller for memory issue
    args.train_output_shape = (args.train_out_size, args.train_out_size)
    # we use the original size of the image (e.g. 96x96 face images) during testing
    args.val_output_shape = (args.val_out_size, args.val_out_size)

    if args.model == 'resnet50':
        model = InsResNet50(pool_size=pool_size)
        desc_dim = {1:64, 2:256, 3:512, 4:1024, 5:2048}
    elif args.model == 'resnet50_half':
        model = InsResNet50(width=0.5, pool_size=pool_size)
        desc_dim = {1:int(64/2), 2:int(256/2), 3:int(512/2), 4:int(1024/2), 5:int(2048/2)}
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
    
    
    # xxx_feat_spectral records the feat dim per layer in hypercol
    # this information is useful to do layer-wise feat normalization in landmark matching
    train_feat_spectral = [] 
    if args.train_use_hypercol:
        for i in range(args.train_layer):
            train_feat_spectral.append(desc_dim[5-i])
    else:
        train_feat_spectral.append(desc_dim[args.train_layer])
    args.train_feat_spectral = train_feat_spectral

    val_feat_spectral = []
    if args.val_use_hypercol:
        for i in range(args.val_layer):
            val_feat_spectral.append(desc_dim[5-i])
    else:
        val_feat_spectral.append(desc_dim[args.val_layer])
    args.val_feat_spectral = val_feat_spectral
    

    # load pretrained moco 
    if args.trained_model_path != 'none':
        print('==> loading pre-trained model')
        ckpt = torch.load(args.trained_model_path, map_location='cpu')
        model.load_state_dict(ckpt['model'], strict=True)
        print("==> loaded checkpoint '{}' (epoch {})".format(
                            args.trained_model_path, ckpt['epoch']))
        print('==> done')
    else:
        print('==> use randomly initialized model')


    # Define feature distiller, set pretrained model to eval mode
    if args.feat_distill:
        model.eval()
        assert np.sum(train_feat_spectral) == np.sum(val_feat_spectral)
        feat_distiller = FeatDistiller(np.sum(val_feat_spectral), 
                                        kernel_size=args.kernel_size,
                                        mode=args.distill_mode,
                                        out_dim = args.out_dim,
                                        softargmax_mul=args.softargmax_mul)
        feat_distiller = nn.DataParallel(feat_distiller)
        feat_distiller.train()
        print('Feature distillation is used: kernel_size:{}, mode:{}, out_dim:{}'.format(
                args.kernel_size, args.distill_mode, args.out_dim))
        feat_distiller = feat_distiller.cuda()
    else:
        feat_distiller = None


    #  evaluate feat distiller on landmark matching, given pretrained moco and feature distiller
    model = model.cuda()
    if args.evaluation_mode:
        if args.feat_distill:
            print("==> use pretrained feature distiller ...")
            feat_ckpt = torch.load(args.trained_feat_model_path, map_location='cpu') 
            # in below, feat_distiller is misspelt, but to use pretrained model, I keep it.
            feat_distiller.load_state_dict(feat_ckpt['feat_disiller'], strict=False)
            print("==> loaded checkpoint '{}' (epoch {})".format(
                                args.trained_feat_model_path, feat_ckpt['epoch']))
            same_err, diff_err = validate(test_loader, model, args, 
                                        feat_distiller=feat_distiller, 
                                        visualization=args.visualize_matching)
        else:
            print("==> use hypercolumn ...")
            same_err, diff_err = validate(test_loader, model, args, 
                                        feat_distiller=None, 
                                        visualization=args.visualize_matching)
        exit()


    ## define optimizer for feature distiller  
    if not args.adam:
        if not args.feat_distill:
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.SGD(feat_distiller.parameters(),
                                        lr=args.learning_rate,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
    else:
        if not args.feat_distill:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.learning_rate,
                                         betas=(args.beta1, args.beta2),
                                         weight_decay=args.weight_decay,
                                         eps=1e-8,
                                         amsgrad=args.amsgrad)
        else:
            optimizer = torch.optim.Adam(feat_distiller.parameters(),
                                         lr=args.learning_rate,
                                         betas=(args.beta1, args.beta2),
                                         weight_decay=args.weight_decay,
                                         eps=1e-8,
                                         amsgrad=args.amsgrad)



    # set lr scheduler
    if args.cosine: # we use cosine scheduler by default
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3) * 0.1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)
    elif args.multistep:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)
    cudnn.benchmark = True

    # report the performance of hypercol on landmark matching tasks
    print("==> Testing of initial model on validation set...")
    same_err, diff_err = validate(val_loader, model, args, feat_distiller=None)
    print("==> Testing of initial model on test set...")
    same_err, diff_err = validate(test_loader, model, args, feat_distiller=None)
    
    # training loss for feature projector
    criterion = dense_corr_loss

    # training feature distiller
    for epoch in range(1, args.epochs + 1):
        if args.cosine or args.multistep:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args, optimizer)

        print("==> training ...")
        time1 = time.time()
        train_loss = train_point_contrast(epoch, train_loader, model, criterion, optimizer, args, 
                                            feat_distiller=feat_distiller)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}, train_loss {:.4f}'.format(epoch, 
                                            time2 - time1, train_loss))
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)


        print("==> validation ...")
        val_same_err, val_diff_err = validate(val_loader, model, args, 
                                            feat_distiller=feat_distiller)


        print("==> testing ...")
        test_same_err, test_diff_err = validate(test_loader, model, args, 
                                            feat_distiller=feat_distiller)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'feat_disiller': feat_distiller.state_dict(),
                'val_error': [val_same_err, val_diff_err],
                'test_error': [test_same_err, test_diff_err],
            }
            save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving regular model!')
            torch.save(state, save_name)

            if val_diff_err < best_error:
                best_error = val_diff_err
                save_name = 'best.pth'
                save_name = os.path.join(args.save_folder, save_name)
                print('saving best model! val_same: {} val_diff: {} test_same: {} test_diff: {}'.format(val_same_err, val_diff_err, test_same_err, test_diff_err))
                torch.save(state, save_name)


def set_lr(optimizer, lr):
    """
    set the learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_point_contrast(epoch, 
                         train_loader, 
                         model,  # pretrained moco
                         criterion, # dense correspondence loss by default
                         optimizer, 
                         opt, 
                         feat_distiller=None):
    """
    one epoch training
    """
    if feat_distiller is None:
        model.train()
    else:
        model.eval()
        feat_distiller.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (input, _, _, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(opt.gpu, non_blocking=True)
        input = input.float()
        
        # ===================forward=====================

        if feat_distiller is not None:
            with torch.no_grad():
                feat = model(input, opt.train_layer, opt.train_use_hypercol, 
                             opt.train_output_shape) 
                feat.detach()
            feat = feat_distiller(feat)
            train_feat_spectral = [opt.out_dim]
        else:
            feat = model(input, opt.train_layer, opt.train_use_hypercol, opt.train_output_shape) 
            train_feat_spectral = opt.train_feat_spectral

        loss = criterion(feat, input.size(), opt, train_feat_spectral)

        if idx == 0:
            print('Layer:{0}, shape of input:{1}, feat:{2}'.format(
                            opt.train_layer, input.size(), feat.size()))

        losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def dense_corr_loss(feat, input_size, opt, feat_spectral, pow=0.5, normalize_vectors=True):
    # feat_spectral is a list of dimensions of features from different layers
    
    B, C, H, W = input_size
    b, c, h, w = feat.size()
    device = feat.device
    stride = H // h
    
    with torch.no_grad():
        yyxx = tps.spatial_grid_unnormalized(H, W).to(device)
        diff = yyxx[::stride, ::stride, None, None, :] - yyxx[None, None, ::stride, ::stride, :]
        diff = (diff * diff).sum(4).sqrt()
        diff = diff.pow(pow)
    
    loss = 0.
    for bb in range(b):
        f1 = feat[bb].reshape(c, h*w)
        if normalize_vectors: 
            f1 = layer_wise_normalize(f1, feat_spectral)
        corr = torch.matmul(f1.t(), f1)
        corr = corr.reshape(h, w, h, w)         

        smcorr = F.softmax(corr.reshape(h, w, -1) * opt.temperature, dim=2).reshape(corr.shape)
        L = diff * smcorr
        loss += L.sum()
    
    return loss / (h * w * b)


def validate(val_loader, model, opt, feat_distiller=None, visualization=False):

    torch.manual_seed(1)
    np.random.seed(1)
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    if feat_distiller is not None:
        feat_distiller.eval()

    same_errs = []
    diff_errs = []

    with torch.no_grad():
        end = time.time()
        for idx, (input, _, target, index) in enumerate(val_loader):

            # get A1, A2, B1, B2 
            # get feats for each images
            input = input.cuda(opt.gpu, non_blocking=True)
            input = input.float()
            target = target.cuda(opt.gpu, non_blocking=True)

            # extract dense descriptors
            x1, x2 = torch.split(input, [3, 3], dim = 1)
            feat_x1 = model(x1, opt.val_layer, opt.val_use_hypercol, opt.val_output_shape)
            feat_x1 = feat_x1.detach()
            feat_x2 = model(x2, opt.val_layer, opt.val_use_hypercol, opt.val_output_shape)
            feat_x2 = feat_x2.detach()
            val_feat_spectral = opt.val_feat_spectral

            if feat_distiller is not None:
                feat_x1 = feat_distiller(feat_x1)
                feat_x2 = feat_distiller(feat_x2)
                val_feat_spectral = [opt.out_dim]

            # images
            im_source = x1[0]
            im_same = x2[0]
            im_diff = x2[1]

            # we use batchsize=2
            feat_source = feat_x1[0] # C*H*W
            feat_same = feat_x2[0]
            feat_diff = feat_x2[1]
            
            # feature normalization
            # We always do the layer wise normalization for hypercolumn in landmark matching
            # otherwise, the performance drops 
            if True: 
                fsrc = layer_wise_normalize(feat_source, val_feat_spectral)
                fsame = layer_wise_normalize(feat_same, val_feat_spectral)
                fdiff = layer_wise_normalize(feat_diff, val_feat_spectral)
            else:
                fsrc = feat_source.clone()
                fsame = feat_same.clone()
                fdiff = feat_diff.clone()

            kp1, kp2 = torch.split(target, [5, 5], dim = 1)
            kp_source = kp1[0]
            kp_same = kp2[0]
            kp_diff = kp2[1]

            B, C, imH, imW = x1.size()
            B, C, featH, featW = feat_x1.size()
            
            if idx == 0:
                print('image shape: {}; feature shape: {}'.format(x1.size(), feat_x1.size()))

            same_match = []
            diff_match = []
            # get the matching and compute the pixel error
            for ki, kp in enumerate(kp_source):
                # normalized keypoints 
                x, y = kp.cpu().numpy()
                gt_same_x, gt_same_y = kp_same[ki].cpu().numpy()
                gt_diff_x, gt_diff_y = kp_diff[ki].cpu().numpy()
                
                same_x, same_y = find_descriptor(x, y, fsrc, fsame)
                err = compute_pixel_err(pred_x=same_x,
                                        pred_y=same_y, 
                                        gt_x=gt_same_x, 
                                        gt_y=gt_same_y, 
                                        insize=featH)
                same_errs.append(err)
                same_match.append([same_y, same_x])

                diff_x, diff_y = find_descriptor(x, y, fsrc, fdiff)
                err = compute_pixel_err(pred_x=diff_x,
                                        pred_y=diff_y, 
                                        gt_x=gt_diff_x, 
                                        gt_y=gt_diff_y, 
                                        insize=featH)
                diff_errs.append(err)
                diff_match.append([diff_y, diff_x])
            
            if visualization:
                # visualize the matching for debugging
                target_outpath = os.path.join(opt.vis_path, str(idx))
                if not os.path.exists(target_outpath):
                    os.makedirs(target_outpath)
                plot_images(im_source, kp_source[:, [1, 0]].cpu().numpy(), 
                                        os.path.join(target_outpath, '1_source.png')) 
                plot_images(im_same, kp_same[:, [1, 0]].cpu().numpy(), 
                                        os.path.join(target_outpath, '2_tsame_gt.png')) 
                plot_images(im_same, same_match, os.path.join(target_outpath, '3_tsame.png')) 
                plot_images(im_diff, kp_diff[:, [1, 0]].cpu().numpy(), 
                                        os.path.join(target_outpath, '4_tdiff_gt.png')) 
                plot_images(im_diff, diff_match, os.path.join(target_outpath, '5_tdiff.png')) 

            batch_time.update(time.time() - end)
            end = time.time()
        print('Same_error {same_mean_err:.3f}\t'
              'Diff_error {diff_mean_err:.3f}'.format(
                    same_mean_err=np.mean(same_errs),
                    diff_mean_err=np.mean(diff_errs)))
    return np.mean(same_errs), np.mean(diff_errs)


def layer_wise_normalize(feat, feat_spectral):
    # we normalize features from different layers seperately for the hypercol representations 
    feat_out = []
    i_prev = 0
    for feat_dim in feat_spectral:
        i = i_prev + feat_dim
        feat_out.append(F.normalize(feat[i_prev: i], p=2, dim=0))
        i_prev = i
    feat_out = torch.cat(feat_out, 0)
    return feat_out


def compute_pixel_err(pred_x, pred_y, gt_x, gt_y, insize):
    # this metric follows DVE's implementations: 
    # https://github.com/jamt9000/DVE/blob/master/test_matching.py
    # we normalize the coordinates from [-1, 1] to [0, 1]
    canonical_sz = 70 
    scale = canonical_sz
    pred_x = (pred_x + 1.) / 2. * (scale - 1)
    pred_y = (pred_y + 1.) / 2. * (scale - 1)
    gt_x = (gt_x + 1.) / 2. * (scale - 1)
    gt_y = (gt_y + 1.) / 2. * (scale - 1)
    return np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)


def find_descriptor(x, y, source_descs, target_descs):
    # input and output of this function are both normalized coors
    C, H, W = source_descs.shape
    x = int(np.round((x + 1.) / 2. * (W - 1)))
    y = int(np.round((y + 1.) / 2. * (H - 1)))
    x = min(W - 1, max(x, 0))
    y = min(H - 1, max(y, 0))
    query_desc = source_descs[:, y, x]
    corr = torch.matmul(query_desc.reshape(-1, C), target_descs.reshape(C, H * W))
    maxidx = corr.argmax()
    grid = tps.spatial_grid_unnormalized(H, W).reshape(-1, 2)
    y, x = grid[maxidx]
    x_norm = 2. * x.item() / (W - 1) - 1 # normalize to [-1, 1]
    y_norm = 2. * y.item() / (H - 1) - 1
    return x_norm, y_norm


def plot_images(image, points, path):
    C, H, W = image.size()
    points = np.array(points)
    points[:, 0] = (points[:, 0] + 1.) / 2. * (H - 1)
    points[:, 1] = (points[:, 1] + 1.) / 2. * (W - 1)
    fig = plt.figure()
    fig.set_size_inches(1., 1, forward = False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(norm_range(image).permute(1, 2, 0).cpu().numpy())
    _cmap = plt.cm.get_cmap('gist_rainbow')
    K = len(points)
    colors = [np.array(_cmap(i)[:3]) for i in np.arange(0,1,1/K)]
    for i, point in enumerate(points):
        ax.scatter(point[1], point[0], c=[colors[i]], marker='.')
    plt.savefig(path, dpi=2*image.shape[1])
    plt.close()


if __name__ == '__main__':
    main()
