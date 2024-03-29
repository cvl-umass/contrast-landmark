# ContrastLandmark

This repository is a PyTorch implementation of <i>On Equivariant and Invariant Learning of Object Landmark Representations</i> by Zezhou
Cheng, Jong-Chyi Su, Subhransu Maji. ICCV 2021.

[[arXiv]](https://arxiv.org/abs/2006.14787v2) [[Project page]](https://people.cs.umass.edu/~zezhoucheng/contrastive_landmark/)  [[Poster]](https://www.dropbox.com/s/5imi8e6m6d895kp/ContrastLandmark_iccv21_poster.pdf?dl=0) [[Supplementary material]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Cheng_On_Equivariant_and_ICCV_2021_supplemental.pdf)


## Installation

The implementation is based on
[DVE](https://github.com/jamt9000/DVE/tree/master/misc/datasets)
[Thewlis et al. ICCV 2019] and
[CMC](https://github.com/HobbitLong/CMC) [Tian et al. 2019]. (Dependencies: tensorboard-logger, pytorch=1.4.0, torchfile)

To install: 
```
conda env create -f environment.yml
conda activate ContrastLandmark
```

## Datasets 

#### Human faces
* Please follow the instruction from [DVE](https://github.com/jamt9000/DVE/tree/master/misc/datasets) to download the datasets. 

#### Birds
* iNaturalist Aves 2017 for training. [[source images](https://github.com/visipedia/inat_comp/tree/master/2017)] [[100K image list](https://people.cs.umass.edu/~zezhoucheng/contrastive_landmark/datasets/inat_aves_100K.txt)]
* CUB dataset for evaluation. [[source images](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)] [[train/val/test set](https://people.cs.umass.edu/~zezhoucheng/contrastive_landmark/datasets/cub_filelist.zip)]

## Experiments

### Training 

#### Stage 1: invariant representation learning

* CelebA 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco.py --batch_size 256 --num_workers 12 --nce_k 4096 --cosine  --epochs 800 --model resnet50 --image_crop 20 --image_size 136 --model_name moco_CelebA --model_path /path/to/save/model --dataset CelebA --data_folder datasets/celeba
```
* iNat Aves
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_moco.py --batch_size 256 --num_workers 12 --nce_k 4096 --cosine  --epochs 800 --model resnet50 --image_crop 0 --image_size 96 --model_name moco_InatAve --model_path /path/to/save/model --dataset InatAve --imagelist /path/to/imagelist/inat_train_100K.txt
```

#### Stage 2: equivariant representation projection

* CelebA 

```
CUDA_VISIBLE_DEVICES=0,1 python train_feature_projector.py --model resnet50 --feat_distill --image_crop 20 --image_size 136 --train_layer 4 --val_layer 4 --trained_model_path /path/to/pretrained_moco --adam --epochs 10 --cosine --batch_size 32 --log_path /path/to/logfile.log --model_name feature_projector --model_path /path/to/save/checkpoint --train_use_hypercol --val_use_hypercol --vis_path /path/to/save/visualization --train_out_size 24 --val_out_size 96 --distill_mode softmax --kernel_size 1 --out_dim 128 --softargmax_mul 7. --temperature 7. 
```

**Note**: 
* `--train_layer 4 --val_layer 4 --train_use_hypercol --val_use_hypercol`: use hypercolumn representations (which consists of features from 4 intermediate layers) as the input to the feature projector;
* To visualize the landmark matching, add `--visualize_matching --vis_path /path/to/save/visualization` to the above command; 
* `--out_dim 128 --softargmax_mul 7. --temperature 7.`: project hypercolumn to 128 dimensional space. We use `--softargmax_mul 7. --temperature 7.` for `--out_dim 128` or `--out_dim 256`, and `--softargmax_mul 6.5 --temperature 8.` for `--out_dim 64`. These hyperparameters are searched on a validation set.
* We provide the logs of training feature projectos for ResNet50-half as a reference: [[training logs]](https://www.dropbox.com/sh/6lmi7y0cmg3abws/AAB-NBMaGGjEz_htGdpecSpGa?dl=0)


### Evaluation: 

#### 1. Landmark regression

##### Face benchmarks (CelebA → AFLW)

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

##### Bird benchmarks (iNat → CUB)

```
CUDA_VISIBLE_DEVICES=0,1 python eval_animal.py --model resnet50 --num_workers 8 --layer 4 --trained_model_path /path/to/pretrainedMoCo --learning_rate 0.01 --weight_decay 0.005 --adam --epochs 2000 --cosine --batch_size 32 --log_path /path/to/logfile --dataset CUB --model_name CUB_regressor --model_path /path/to/save/regressor --image_crop 0 --image_size 96 --imagelist /path/to/trainlist/train.txt --use_hypercol
```

**Note**: check out [`data_loaders_animal.py`](./data_loader/data_loaders_animal.py), place the annotation files (train.dat, val.data) and train/val/test text files under `./datasets/CUB-200-2011`. About hyperparameter settings on bird benchmarks, if the number of annotations is smaller or equal to 100 (e.g. 10,
50, 100), lr=0.01 and weight decay=0.05 for ResNet18, ResNet50, and DVE; if more annotations (e.g. 250, 500, 1241) are available, lr=0.01 and weight decay=0.005 for ResNet18 and ResNet50, but lr=0.01 and weight decay=0.0005 for DVE (because DVE has much better performance with WD=0.0005 than WD=0.05 or 0.005)

#### 2. Landmark matching 

* CelebA 

```
CUDA_VISIBLE_DEVICES=0,1 python train_feature_projector.py --model resnet50 --feat_distill --image_crop 20 --image_size 136 --train_layer 4 --val_layer 4 --trained_model_path /path/to/pretrained_moco  --log_path /path/to/logfile.log --model_name feature_projector --model_path /path/to/save/tmpfile --train_use_hypercol --val_use_hypercol  --train_out_size 24 --val_out_size 96 --distill_mode softmax --kernel_size 1 --out_dim 128 --softargmax_mul 7. --temperature 7. --evaluation_mode --trained_feat_model_path /path/to/pretrained-feature-projector --visualize_matching --vis_path /path/to/save/visualization
```

**Note**: 
* You could assign any strings to some arguments: `--model_name feature_projector`
* `--visualize_matching --vis_path /path/to/save/visualization`: visualize the landmark matching results, remove `--visualize_matching` to turn off the visualization
* To test the performance of hypercolumn without feature projection, remove `--feat_distill`
* Modify `--out_dim 128 --softargmax_mul 7. --temperature 7.` accordingly when testing other feature projection dimensions (e.g. 64, 256). `--softargmax_mul 7. --temperature 7.` for `--out_dim 256`; `--softargmax_mul 6.5 --temperature 8.` for `--out_dim 64`.
* See [examples](./examples/example_test.sh) on how to run the landmark matching with hypercolumn or projected features.

## Pretrained models

### Download the pretrained models

* Contrastively learning models:
 1. Celeb:
[[MoCo-ResNet18-CelebA](https://www.dropbox.com/sh/f9act9d7wlspm3c/AAACHwe9BZVKFQkokvGvhYrKa?dl=0)]
[[MoCo-ResNet50-CelebA](https://www.dropbox.com/sh/jys3jerh0utxr49/AAAEzPJ3ZN4XLUmc4pmXEytFa?dl=0)]
[[MoCo-ResNet50-CelebA-In-the-Wild](https://www.dropbox.com/s/6y3ns9cqbpodj69/ckpt_epoch_800_resnet50_celeba_wild.pth?dl=0)]
 2. iNat Aves:
[[MoCo-ResNet18-iNat](https://www.dropbox.com/sh/vf6l9t4e5rbzaf1/AAAgeIcD-TjYHw9B41LcIMbTa?dl=0])] 
[[MoCo-ResNet50-iNat](https://www.dropbox.com/sh/g1folefnc351eyf/AAD5bmVrvNesTY8Put95WIV0a?dl=0)] 
[[DVE-Hourglass-iNat](https://www.dropbox.com/sh/hmks0is2v67zn5x/AABS4cxUlH-oVv8zH8pzgLzSa?dl=0)]

* Linear-regressor: 
[[Face benchmarks](https://www.dropbox.com/sh/cx3m6s4soompt9r/AADDDPeYeOtvCazN7x53vXiFa?dl=0)] 
[[Bird benchmarks](https://www.dropbox.com/sh/jqn6umci2vlngkb/AAB5740XNLzyAQSPohjkXUOOa?dl=0)]

**Note**: On face benchmarks, the numbers in Table 1 in the main text are reported at 120th, 45th, 80th epoch for MAFL, AFLW and 300W. The epoch is indexing from 0. However, the index was starting from 1 when we saved the model. This leads to different scores with the saved model from these in Table 1 (either slightly better or slightly worse).  

* Pretrained feature projector [[Feature projectors](https://www.dropbox.com/sh/ygq7qe24p4pl98l/AAA4PqfBl5327M3rozgfqM4fa?dl=0)]

The feature projectors are trained under different network architectures (e.g. ResNet18, ResNet50, ResNet50-half, etc.) and pretraining methods (e.g. MoCo, ImageNet, Random Init etc.). These settings corresponds to Table 4 and Table 5 in the supplementary material. 


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
@inproceedings{cheng2021equivariant,
title={On Equivariant and Invariant Learning of Object Landmark Representations},
author={Cheng, Zezhou and Su, Jong-Chyi and Maji, Subhransu},
booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
pages={9897--9906},
year={2021}
}
```

