"""
evaluating MoCo in the task of landmark regression 
on animal datasets (e.g. CUB)
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

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')

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
    parser.add_argument('--image_crop', type=int, default=0, help='image pre-crop') # image preprocessing
    parser.add_argument('--image_size', type=int, default=96, help='image size') # image preprocessing

    # dataset
    parser.add_argument('--dataset', type=str, default='CUB')
    parser.add_argument('--imagelist', type=str, default=None)

    # model path and name  
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--model_path', type=str)

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

    opt = parser.parse_args()


    num_annotated_points = {
        "CUB": 15
    }

    opt.data_folder = './datasets/CUB_200_2011'
    opt.save_path = opt.model_path
    opt.tb_path = '%s_tensorboard' % opt.model_path
    opt.num_points = num_annotated_points[opt.dataset]

    Tee(opt.log_path, 'a')

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.save_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def main():

    global best_val_pck, best_test_pck
    best_val_pck = 0
    best_test_pck = 0

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    train_dataset = getattr(module_data, args.dataset)(
                                         args.data_folder,
                                         train=True,
                                         pair_image=False, 
                                         imwidth=args.image_size, 
                                         crop=args.image_crop,
                                         imagelist=args.imagelist)
    val_dataset   = getattr(module_data, args.dataset)(
                                         args.data_folder,
                                         val=True, 
                                         pair_image=False, 
                                         imwidth=args.image_size, 
                                         crop=args.image_crop)
    test_dataset   = getattr(module_data, args.dataset)(
                                         args.data_folder,
                                         test=True, 
                                         pair_image=False, 
                                         imwidth=args.image_size, 
                                         crop=args.image_crop)

    print('Number of training images: %d' % len(train_dataset))
    print('Number of validation images: %d' % len(val_dataset))
    print('Number of test images: %d' % len(test_dataset))

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # create model and optimizer

    input_size = args.image_size - 2 * args.image_crop
    pool_size = int(input_size / 2**5) # 96x96 --> 3; 160x160 --> 5; 224x224 --> 7;
    args.output_shape = (48,48)
    args.boxsize=48

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
    
    regressor =  IntermediateKeypointPredictor(feat_dim, num_annotated_points=args.num_points, 
                                                num_intermediate_points=50, 
                                                softargmax_mul = 100.0)

    print('==> loading pre-trained model')
    ckpt = torch.load(args.trained_model_path, map_location='cpu')
    if args.model == 'hourglass':
        model.load_state_dict(clean_state_dict(ckpt["state_dict"])) 
    else:
        model.load_state_dict(ckpt['model'], strict=False)
    print("==> loaded checkpoint '{}' (epoch {})".format(args.trained_model_path, ckpt['epoch']))
    print('==> done')

    model = model.cuda()
    regressor = regressor.cuda()

    criterion = selected_regression_loss

    if not args.adam:
        optimizer = torch.optim.SGD(regressor.parameters(),
                                    lr=args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(regressor.parameters(),
                                     lr=args.learning_rate,
                                     betas=(args.beta1, args.beta2),
                                     weight_decay=args.weight_decay,
                                     eps=1e-8,
                                     amsgrad=args.amsgrad)
    model.eval()
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            regressor.load_state_dict(checkpoint['regressor'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_val_pck = checkpoint['best_val_pck']
            best_test_pck = checkpoint['best_test_pck']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            if 'opt' in checkpoint.keys():
                # resume optimization hyper-parameters
                print('=> resume hyper parameters')
                if 'bn' in vars(checkpoint['opt']):
                    print('using bn: ', checkpoint['opt'].bn)
                if 'adam' in vars(checkpoint['opt']):
                    print('using adam: ', checkpoint['opt'].adam)
                if 'cosine' in vars(checkpoint['opt']):
                    print('using cosine: ', checkpoint['opt'].cosine)
                args.learning_rate = checkpoint['opt'].learning_rate
                # args.lr_decay_epochs = checkpoint['opt'].lr_decay_epochs
                args.lr_decay_rate = checkpoint['opt'].lr_decay_rate
                args.momentum = checkpoint['opt'].momentum
                args.weight_decay = checkpoint['opt'].weight_decay
                args.beta1 = checkpoint['opt'].beta1
                args.beta2 = checkpoint['opt'].beta2
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # set cosine annealing scheduler
    if args.cosine:
        eta_min = args.learning_rate * (args.lr_decay_rate ** 3) * 0.1
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min, -1)
        # dummy loop to catch up with current epoch
        for i in range(1, args.start_epoch):
            scheduler.step()
    elif args.multistep:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 250], gamma=0.1)
        # dummy loop to catch up with current epoch
        for i in range(1, args.start_epoch):
            scheduler.step()

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        if args.cosine or args.multistep:
            scheduler.step()
        else:
            adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        PCK, train_loss = train(epoch, train_loader, model, regressor, criterion, optimizer, args)
        time2 = time.time()
        print('train epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('PCK', PCK, epoch)
        logger.log_value('train_loss', train_loss, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        print("==> validating...")
        val_PCK, val_loss = validate(val_loader, model, regressor, criterion, args)

        logger.log_value('Val_PCK', val_PCK, epoch)
        logger.log_value('val_loss', val_loss, epoch) 

        print("==> testing...")
        test_PCK, test_loss = validate(test_loader, model, regressor, criterion, args)

        logger.log_value('Test_PCK', test_PCK, epoch)
        logger.log_value('test_loss', test_loss, epoch) 

        # save the best model
        if val_PCK >  best_val_pck:
            best_val_pck = val_PCK
            best_test_pck = test_PCK
            state = {
                'opt': args,
                'epoch': epoch,
                'regressor': regressor.state_dict(),
                'best_test_pck': best_test_pck,
                'best_val_pck': best_val_pck,
                'optimizer': optimizer.state_dict(),
            }
            save_name = '{}.pth'.format(args.model)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving best model!')
            torch.save(state, save_name)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'epoch': epoch,
                'regressor': regressor.state_dict(),
                'best_test_pck': best_test_pck,
                'best_val_pck': best_val_pck,
                'optimizer': optimizer.state_dict(),
            }
            save_name = 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch)
            save_name = os.path.join(args.save_folder, save_name)
            print('saving regular model!')
            torch.save(state, save_name)

        logger.log_value('best_val_pck', best_val_pck, epoch) 
        logger.log_value('best_test_pck', best_test_pck, epoch) 

        print('* Best test pck: %.2f; Best val pck: %.2f; Epoch: %d' % (best_test_pck, best_val_pck, epoch))


def set_lr(optimizer, lr):
    """
    set the learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(epoch, train_loader, model, regressor, criterion, optimizer, opt):
    """
    one epoch training
    """

    model.eval()
    regressor.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    PCK = AverageMeter()

    end = time.time()
    for idx, (input, visible, target, index) in enumerate(train_loader): 
        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(opt.gpu, non_blocking=True)
        input = input.float()
        target = target.cuda(opt.gpu, non_blocking=True)
        target = target.float()

        # ===================forward=====================
        with torch.no_grad():
            feat = model(input, opt.layer, opt.use_hypercol, opt.output_shape)
            feat = feat.detach()
        output, _ = regressor(feat)
        loss = criterion(output, target, visible, alpha=10.)
        # print(loss.type(), output.type(), target.type())

        if idx == 0:
            print('Layer:{0}, shape of input:{1}, feat:{2}, output:{3}'.format(opt.layer, 
                                input.size(), feat.size(), output.size()))

        # ic_error = inter_ocular_error(output, target, eyeidxs=opt.eye_idx)
        # the max box size is set to 48, to be consistent with the output size of DVE
        ic_error = calc_pck(output, target, visible, boxsize=opt.boxsize)
        losses.update(loss.item(), input.size(0))
        PCK.update(ic_error, input.size(0))

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
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PCK {PCK.val:.3f}'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, PCK=PCK))
            sys.stdout.flush()

    return PCK.avg, losses.avg


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
            # if opt.gpu is not None:
            input = input.cuda(opt.gpu, non_blocking=True)
            input = input.float()
            target = target.cuda(opt.gpu, non_blocking=True)
            target = target.float()

            # compute output
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

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'PCK {PCK.val:.3f}'.format(
                    idx, len(val_loader), batch_time=batch_time, loss=losses,
                    PCK=PCK))

        print(' * PCK {PCK.avg:.3f}'
              .format(PCK=PCK))

    return PCK.avg, losses.avg


if __name__ == '__main__':
    best_pck = 0
    main()
