#!/bin/bash

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco.py --batch_size 256 --num_workers 12 --nce_k 4096 --cosine  --epochs 800 --model resnet50 --image_crop 20 --image_size 136 --model_name moco_CelebA --model_path ./Pretrained/ --dataset CelebA --data_folder datasets/celeba

#CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco.py --batch_size 256 --num_workers 12 --nce_k 4096 --cosine  --epochs 800 --model resnet50 --image_crop 0 --image_size 96 --model_name moco_InatAve --model_path ./Pretrained/ --dataset InatAve --imagelist ./datasets/inat/train_100000.txt

