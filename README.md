# Unsupervised Discovery of Object Landmarks via Contrastive Learning

![Teaser Image](https://people.cs.umass.edu/~zezhoucheng/contrastive_landmark/figs/fig1.png)

This repository is a PyTorch implementation of <i>Unsupervised
Discovery of Object Landmarks via Contrastive Learning</i> Zezhou
Cheng, Jong-Chyi Su, Subhransu Maji, <u>arXiv:2006.14787, 2020</u>. 

[[arXiv]](https://arxiv.org/abs/2006.14787) [[Project page]](https://people.cs.umass.edu/~zezhoucheng/contrastive_landmark/)  


## Installation

The implementation is based on
[DVE](https://github.com/jamt9000/DVE/tree/master/misc/datasets)
[Thewlis et al. ICCV 2019] and
[CMC](https://github.com/HobbitLong/CMC) [Tian et al. 2019]. (Dependencies: tensorboard-logger, pytorch=1.4.0, torchfile)

To install: 
```
conda create -n ContrastLandmark python=3.7.3 anaconda
source activate ContrastLandmark
conda install pytorch=1.4.0 torchvision -c pytorch
pip install tensorboard-logger
pip install torchfile
```

## Datasets 

### Human faces
* Please follow the instruction from [DVE](https://github.com/jamt9000/DVE/tree/master/misc/datasets) to download the datasets. 

### Birds
* iNaturalist Aves 2017 for training. [[source images](https://github.com/visipedia/inat_comp/tree/master/2017)] [[100K image list](https://people.cs.umass.edu/~zezhoucheng/contrastive_landmark/datasets/inat_aves_100K.txt)]
* CUB dataset for evaluation. [[source images](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)] [[train/val/test set](https://people.cs.umass.edu/~zezhoucheng/contrastive_landmark/datasets/cub_filelist.zip)]

## Experiments

### Train representations using MoCo

* CelebA 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco.py --batch_size 256 --num_workers 12 --nce_k 4096 --cosine  --epochs 800 --model resnet50 --image_crop 20 --image_size 136 --model_name moco_CelebA --model_path /path/to/save/model --dataset CelebA --data_folder datasets/celeba
```
* iNat Aves
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco.py --batch_size 256 --num_workers 12 --nce_k 4096 --cosine  --epochs 800 --model resnet50 --image_crop 0 --image_size 96 --model_name moco_InatAve --model_path /path/to/save/model --dataset InatAve --imagelist /path/to/imagelist/inat_train_100K.txt
```

### Landmark regression evaluation

#### Face benchmarks (CelebA → AFLW)

- Use hypercolumn as representation (`--use_hypercol`) with activation from conv2_x to conv5_x (`--layer 4`)
```
CUDA_VISIBLE_DEVICES=0,1 python eval_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path /path/to/pretrainedMoCo --learning_rate 0.001 --weight_decay 0.0005 --adam --epochs 200 --cosine --batch_size 32 --log_path /path/to/logfile --dataset AFLW_MTFL --model_name AFLW_M_regressor --model_path /path/to/save/regressor --image_crop 20 --image_size 136 --use_hypercol
```

- Limited annotations, e.g., only 50 annotated face images from AFLW_MTFL are available, we use thin-plate spline for data augmentation (`--TPS_aug`)
```
CUDA_VISIBLE_DEVICES=0 python eval_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path /path/to/pretrainedMoCo --learning_rate 0.01 --weight_decay 0.05 --adam --epochs 1000 --cosine --batch_size 32 --log_path /path/to/logfile --dataset AFLW_MTFL --model_name AFLW_M_regressor --model_path /path/to/save/regressor --image_crop 20 --image_size 136 --restrict_annos 50  --repeat --TPS_aug --use_hypercol
```

**Note**: the number of GPUs used to train the linear regressor has
impact on the convergence rate, the possible reason is the batch
normalization is conducted separately on different GPUs. 
We stop the training procedure at 120th, 45th, 80th epoch on MAFL, AFLW, and 300W
benchmarks respectively on 2 GPUs (determined based on our initial
results and kept fixed in our experiments).
However, the stopping points may be suboptimal when you train
the regressor on a different number of GPUs. 

#### Bird benchmarks (iNat → CUB)

```
CUDA_VISIBLE_DEVICES=0,1 python eval_animal.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path /path/to/pretrainedMoCo --learning_rate 0.01 --weight_decay 0.005 --adam --epochs 2000 --cosine --batch_size 32 --log_path /path/to/logfile --dataset CUB --model_name CUB_regressor --model_path /path/to/save/regressor --image_crop 0 --image_size 96 --imagelist /path/to/trainlist/train.txt --use_hypercol
```

**Note**: check out [`data_loaders_animal.py`](./data_loader/data_loaders_animal.py), place the annotation files (train.dat, val.data) and train/val/test text files under `./datasets/CUB-200-2011`. About hyperparameter settings on bird benchmarks, if the number of annotations is smaller or equal to 100 (e.g. 10,
50, 100), lr=0.01 and weight decay=0.05 for ResNet18, ResNet50, and DVE; if more annotations (e.g. 250, 500, 1241) are available, lr=0.01 and weight decay=0.005 for ResNet18 and ResNet50, but lr=0.01 and weight decay=0.0005 for DVE (because DVE has much better performance with WD=0.0005 than WD=0.05 or 0.005)

## Pretrained models

### Download the pretrained models

* Contrastively learning models:
 1. Celeb:
[[MoCo-ResNet18-CelebA](https://www.dropbox.com/sh/f9act9d7wlspm3c/AAACHwe9BZVKFQkokvGvhYrKa?dl=0)]
[[MoCo-ResNet50-CelebA](https://www.dropbox.com/sh/jys3jerh0utxr49/AAAEzPJ3ZN4XLUmc4pmXEytFa?dl=0)]
 2. iNat Aves:
[[MoCo-ResNet18-iNat](https://www.dropbox.com/sh/vf6l9t4e5rbzaf1/AAAgeIcD-TjYHw9B41LcIMbTa?dl=0])] 
[[MoCo-ResNet50-iNat](https://www.dropbox.com/sh/g1folefnc351eyf/AAD5bmVrvNesTY8Put95WIV0a?dl=0)] 
[[DVE-Hourglass-iNat](https://www.dropbox.com/sh/hmks0is2v67zn5x/AABS4cxUlH-oVv8zH8pzgLzSa?dl=0)]

* Linear-regressor: 
[[Face benchmarks](https://www.dropbox.com/sh/cx3m6s4soompt9r/AADDDPeYeOtvCazN7x53vXiFa?dl=0)] 
[[Bird benchmarks](https://www.dropbox.com/sh/jqn6umci2vlngkb/AAB5740XNLzyAQSPohjkXUOOa?dl=0)]

**Note**: On face benchmarks, the numbers in Table 1 in the main text are reported at 120th, 45th, 80th epoch for MAFL, AFLW and 300W. The epoch is indexing from 0. However, the index was starting from 1 when we saved the model. This leads to different scores with the saved model from these in Table 1 (either slightly better or slightly worse).  

### Run pretrained landmark detectors

After downloading the pretrained models, run the following commands to evaluate and visualize the pretrained models.

```
pretrained_MOCO_FACE=./Pretrained/ckpt_epoch_800_resnet50_celeba.pth
pretrained_MOCO_inat=./Pretrained/ckpt_epoch_800_resnet50_inat.pth
pretrained_AFLW_R=./Pretrained/ckpt_epoch_45_resnet50_AFLW_R.pth
pretrained_CUB=./Pretrained/CUB_resnet50.pth
visdir=./Visualization
```

1. Face benchmarks:
```
CUDA_VISIBLE_DEVICES=0 python vis_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_FACE --batch_size 32 --log_path $log_file --dataset AFLW --image_crop 20 --image_size 136 --ckpt_path $pretrained_AFLW_R  --vis_path $visdir --use_hypercol --vis_keypoints
```

1. Bird benchmarks:
```
CUDA_VISIBLE_DEVICES=0 python vis_animal.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_inat --batch_size 32 --log_path $log_file --dataset CUB --image_crop 0 --image_size 96 --ckpt_path $pretrained_CUB  --vis_path $visdir --use_hypercol --vis_keypoints
```

### Visualize the PCA projection of hypercolumn representation

1. Face benchmarks:
```
CUDA_VISIBLE_DEVICES=0 python vis_face.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_FACE --batch_size 32 --log_path $log_file --dataset MAFLAligned --image_crop 20 --image_size 136 --vis_path $visdir --use_hypercol --vis_PCA
```

2. Bird benchmarks:
```
CUDA_VISIBLE_DEVICES=0 python vis_animal.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path $pretrained_MOCO_inat --batch_size 32 --log_path $log_file --dataset CUB --image_crop 0 --image_size 96   --vis_path $visdir --use_hypercol --vis_PCA
```

## Citation
If you use this code for your research, please cite the following papers.

```
@article{cheng2020unsupervised,
title={Unsupervised Discovery of Object Landmarks via Contrastive Learning},
author={Cheng, Zezhou and Su, Jong-Chyi and Maji, Subhransu},
journal={arXiv preprint arXiv:2006.14787},
year={2020}
}
```

