"""
Training MoCo

@author: Yonglong Tian - https://github.com/HobbitLong/CMC
"""

from __future__ import print_function

import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn
import argparse
import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from utils.util import adjust_learning_rate, AverageMeter

from models.resnet import InsResNet50,InsResNet18,InsResNet34,InsResNet101,InsResNet152
from NCE.NCEAverage import MemoryMoCo
from NCE.NCECriterion import NCESoftmaxLoss
from data_loader.data_loaders_face import CelebAPrunedAligned_MAFLVal
from data_loader.data_loaders_animal import InatAve


def parse_option():

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=20, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=18, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.03, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='120,160,200', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--cosine', action='store_true', help='use cosine annealing')

    # image augmentation: random sized cropping 
    parser.add_argument('--crop', type=float, default=0.2, help='minimum crop') # random resized crop
    # image preprocessing (resize and crop)
    parser.add_argument('--image_crop', type=int, default=15, help='image pre-crop') # image preprocessing
    parser.add_argument('--image_size', type=int, default=100, help='image size') # image preprocessing

    # dataset
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['InatAve', 'CelebA'])
    parser.add_argument('--data_folder', type=str, default='./datasets/celeba', help='the path the dataset')
    parser.add_argument('--imagelist', type=str, default=None)

    # model path and name
    parser.add_argument('--model_name', type=str) 
    parser.add_argument('--model_path', type=str) # path to store the models

    # resume
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    # model definition
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet50', 'resnet50x2', 'resnet50x4', 
                                 'resnet18', 'resnet34', 'resnet101', 'resnet152'])

    # loss function
    parser.add_argument('--nce_k', type=int, default=4096)
    parser.add_argument('--nce_t', type=float, default=0.07)
    parser.add_argument('--nce_m', type=float, default=0.5)

    # memory setting
    parser.add_argument('--alpha', type=float, default=0.999, help='exponential moving average weight')

    # GPU setting
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.tb_path = opt.model_path + '_tensorboard'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.method = 'softmax'
    
    opt.model_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.model_folder):
        os.makedirs(opt.model_folder)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    return opt


def moment_update(model, model_ema, m):
    """ model_ema = m * model_ema + (1 - m) model """
    for p1, p2 in zip(model.parameters(), model_ema.parameters()):
        p2.data.mul_(m).add_(1-m, p1.detach().data)


def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


def main():

    args = parse_option()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.dataset == 'CelebA':
        train_dataset = CelebAPrunedAligned_MAFLVal(args.data_folder, 
                                                    train=True, 
                                                    pair_image=True, 
                                                    do_augmentations=True,
                                                    imwidth=args.image_size,
                                                    crop = args.image_crop)
    elif args.dataset == 'InatAve':
        train_dataset = InatAve(args.data_folder,
                                train=True, 
                                pair_image=True, 
                                do_augmentations=True, 
                                imwidth=args.image_size, 
                                imagelist=args.imagelist)
    else:
        raise NotImplementedError('dataset not supported {}'.format(args.dataset))

    print(len(train_dataset))
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

    # create model and optimizer
    n_data = len(train_dataset)

    input_size = args.image_size - 2 * args.image_crop
    pool_size = int(input_size / 2**5) # 96x96 --> 3; 160x160 --> 5; 224x224 --> 7; 

    if args.model == 'resnet50':
        model = InsResNet50(pool_size=pool_size)
        model_ema = InsResNet50(pool_size=pool_size)
    elif args.model == 'resnet50x2':
        model = InsResNet50(width=2, pool_size=pool_size)
        model_ema = InsResNet50(width=2, pool_size=pool_size)
    elif args.model == 'resnet50x4':
        model = InsResNet50(width=4, pool_size=pool_size)
        model_ema = InsResNet50(width=4, pool_size=pool_size)
    elif args.model == 'resnet18':
        model = InsResNet18(width=1, pool_size=pool_size)
        model_ema = InsResNet18(width=1, pool_size=pool_size)
    elif args.model == 'resnet34':
        model = InsResNet34(width=1, pool_size=pool_size)
        model_ema = InsResNet34(width=1, pool_size=pool_size)
    elif args.model == 'resnet101':
        model = InsResNet101(width=1, pool_size=pool_size)
        model_ema = InsResNet101(width=1, pool_size=pool_size)
    elif args.model == 'resnet152':
        model = InsResNet152(width=1, pool_size=pool_size)
        model_ema = InsResNet152(width=1, pool_size=pool_size)
    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    # copy weights from `model' to `model_ema'
    moment_update(model, model_ema, 0)

    # set the contrast memory and criterion
    contrast = MemoryMoCo(128, n_data, args.nce_k, args.nce_t, use_softmax=True).cuda(args.gpu)

    criterion = NCESoftmaxLoss()
    criterion = criterion.cuda(args.gpu)

    model = model.cuda()
    model_ema = model_ema.cuda()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # optionally resume from a checkpoint
    args.start_epoch = 1
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            # checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            contrast.load_state_dict(checkpoint['contrast'])
            model_ema.load_state_dict(checkpoint['model_ema'])
            print("=> loaded successfully '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # tensorboard
    logger = tb_logger.Logger(logdir=args.tb_folder, flush_secs=2)

    # routine
    for epoch in range(args.start_epoch, args.epochs + 1):

        adjust_learning_rate(epoch, args, optimizer)
        print("==> training...")

        time1 = time.time()
        loss, prob = train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, args)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # tensorboard logger
        logger.log_value('ins_loss', loss, epoch)
        logger.log_value('ins_prob', prob, epoch)
        logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # save model
        if epoch % args.save_freq == 0:
            print('==> Saving...')
            state = {
                'opt': args,
                'model': model.state_dict(),
                'contrast': contrast.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }
            state['model_ema'] = model_ema.state_dict()
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
            # help release GPU memory
            del state

        # saving the model
        print('==> Saving...')
        state = {
            'opt': args,
            'model': model.state_dict(),
            'contrast': contrast.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        state['model_ema'] = model_ema.state_dict()
        save_file = os.path.join(args.model_folder, 'current.pth')
        torch.save(state, save_file)
        if epoch % args.save_freq == 0:
            save_file = os.path.join(args.model_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)
        # help release GPU memory
        del state
        torch.cuda.empty_cache()


def train_moco(epoch, train_loader, model, model_ema, contrast, criterion, optimizer, opt):
    """
    one epoch training for instance discrimination
    """

    model.train()
    model_ema.eval()

    def set_bn_train(m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model_ema.apply(set_bn_train)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    prob_meter = AverageMeter()

    end = time.time()
    for idx, (inputs, _, _, index) in enumerate(train_loader):
        data_time.update(time.time() - end)

        bsz = inputs.size(0)

        inputs = inputs.float()
        if opt.gpu is not None:
            inputs = inputs.cuda(opt.gpu, non_blocking=True)
        else:
            inputs = inputs.cuda()
        index = index.cuda(opt.gpu, non_blocking=True)

        # ===================forward=====================
        x1, x2 = torch.split(inputs, [3, 3], dim=1)

        # ids for ShuffleBN
        shuffle_ids, reverse_ids = get_shuffle_ids(bsz)

        feat_q = model(x1)
        with torch.no_grad():
            x2 = x2[shuffle_ids]
            feat_k = model_ema(x2)
            feat_k = feat_k[reverse_ids]

        out = contrast(feat_q, feat_k)

        loss = criterion(out)
        prob = out[:, 0].mean()

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        loss_meter.update(loss.item(), bsz)
        prob_meter.update(prob.item(), bsz)

        moment_update(model, model_ema, opt.alpha)

        torch.cuda.synchronize()
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'prob {prob.val:.3f} ({prob.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_meter, prob=prob_meter))
            print(out.shape)
            sys.stdout.flush()

    return loss_meter.avg, prob_meter.avg


if __name__ == '__main__':
    main()
