#!/bin/bash

# CUDA_VISIBLE_DEVICES=0,1 python eval_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path ./Pretrained/ckpt_epoch_800_resnet50_celeba.pth --learning_rate 0.001 --weight_decay 0.0005 --adam --epochs 150 --cosine --batch_size 32 --log_path Logs/regress_AFLW_M --dataset AFLW_MTFL --model_name AFLW_M_regressor --model_path ./Pretrained/ --image_crop 20 --image_size 136 --use_hypercol

# CUDA_VISIBLE_DEVICES=0,1 python eval_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path ./Pretrained/ckpt_epoch_800_resnet50_celeba.pth --learning_rate 0.001 --weight_decay 0.0005 --adam --epochs 50 --cosine --batch_size 32 --log_path Logs/regress_AFLW_M_conv4 --dataset AFLW_MTFL --model_name AFLW_M_regressor --model_path ./Pretrained/ --image_crop 20 --image_size 136

#CUDA_VISIBLE_DEVICES=0 python eval_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path ./Pretrained/ckpt_epoch_800_resnet50_celeba.pth --learning_rate 0.01 --weight_decay 0.05 --adam --epochs 50 --cosine --batch_size 32 --log_path Logs/regress_AFLW_M_limited --dataset AFLW_MTFL --model_name AFLW_M_regressor_limited --model_path ./Pretrained/ --image_crop 20 --image_size 136 --restrict_annos 50  --repeat --TPS_aug --use_hypercol

#CUDA_VISIBLE_DEVICES=0 python eval_animal.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path ./Pretrained/ckpt_epoch_800_resnet50_inat.pth --learning_rate 0.001 --weight_decay 0.0005 --adam --epochs 2000 --cosine --batch_size 32 --log_path Logs/regress_CUB --dataset CUB --model_name CUB_regressor --model_path ./Pretrained/ --image_crop 0 --image_size 96 --imagelist ./datasets/CUB_200_2011/split/train.txt --use_hypercol

# sbatch -p titanx-short --gres=gpu:1 --mem=40000 -o Logs/%J.out example_test.sh

