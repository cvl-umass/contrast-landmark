#!/bin/bash

mkdir Pretrained
mkdir Logs
mkdir Visualization

pretrained_MOCO_FACE=./Pretrained/ckpt_epoch_800_resnet50_celeba.pth
pretrained_MOCO_inat=./Pretrained/ckpt_epoch_800_resnet50_inat.pth
pretrained_DVE_inat=./Pretrained/ckpt_epoch100_DVE_inat.pth
pretrained_AFLW_M=./Pretrained/ckpt_epoch_45_resnet50_AFLW_M.pth
pretrained_AFLW_R=./Pretrained/ckpt_epoch_45_resnet50_AFLW_R.pth
pretrained_MAFL=./Pretrained/ckpt_epoch_120_resnet50_MAFL.pth
pretrained_300W=./Pretrained/ckpt_epoch_80_resnet50_300W.pth
pretrained_CUB=./Pretrained/CUB_resnet50.pth
log_file=./Logs/test_pretrained_moco_on_AFLW_M

CUDA_VISIBLE_DEVICES=4 python vis_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_FACE --batch_size 32 --log_path $log_file --dataset MAFLAligned --image_crop 20 --image_size 136 --ckpt_path $pretrained_MAFL  --vis_path Visualization --use_hypercol --vis_keypoints
CUDA_VISIBLE_DEVICES=4 python vis_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_FACE --batch_size 32 --log_path $log_file --dataset AFLW_MTFL --image_crop 20 --image_size 136 --ckpt_path $pretrained_AFLW_M  --vis_path Visualization --use_hypercol --vis_keypoints
CUDA_VISIBLE_DEVICES=4 python vis_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_FACE --batch_size 32 --log_path $log_file --dataset AFLW --image_crop 20 --image_size 136 --ckpt_path $pretrained_AFLW_R  --vis_path Visualization --use_hypercol --vis_keypoints
CUDA_VISIBLE_DEVICES=4 python vis_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_FACE --batch_size 32 --log_path $log_file --dataset ThreeHundredW --image_crop 20 --image_size 136 --ckpt_path $pretrained_300W  --vis_path Visualization --use_hypercol --vis_keypoints
CUDA_VISIBLE_DEVICES=4 python vis_animal.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_inat --batch_size 32 --log_path $log_file --dataset CUB --image_crop 0 --image_size 96 --ckpt_path $pretrained_CUB  --vis_path Visualization --use_hypercol --vis_keypoints

# PCA
CUDA_VISIBLE_DEVICES=4 python vis_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_FACE --batch_size 32 --log_path $log_file --dataset MAFLAligned --image_crop 20 --image_size 136 --ckpt_path  --vis_path Visualization --use_hypercol --vis_PCA
CUDA_VISIBLE_DEVICES=4 python vis_animal.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_inat --batch_size 32 --log_path $log_file --dataset CUB --image_crop 0 --image_size 96  --vis_path Visualization --use_hypercol --vis_PCA
CUDA_VISIBLE_DEVICES=1 python vis_animal.py --model hourglass --num_workers 8 --layer 4 --trained_model_path $pretrained_DVE_inat --batch_size 32 --log_path $log_file --dataset CUB --image_crop 0 --image_size 96  --vis_path Visualization --use_hypercol --vis_PCA

# sbatch -p titanx-short --gres=gpu:1 --mem=40000 -o Logs/%J.out example_pretrained.sh
