# DiscoNet: Learning Distilled Collaboration Graph for Multi-Agent Perception [NeurIPS 2021]

[Yiming Li](https://scholar.google.com/citations?user=i_aajNoAAAAJ), [Shunli Ren](https://github.com/ShunliRen), [Pengxiang Wu](https://scholar.google.com/citations?user=MXLs7GcAAAAJ&hl=en), [Siheng Chen](https://scholar.google.com/citations?user=W_Q33RMAAAAJ&hl=en), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ), [Wenjun Zhang](https://www.researchgate.net/profile/Wenjun-Zhang-29)

**Learn a digraph with matrix-valued edge weight for multi-agent perception.**

<p align="center"><img src='img.png' align="center" height="400px"> </p>

[**ArXiv: Learning Distilled Collaboration Graph for Multi-Agent Perception**](https://arxiv.org/abs/2103.15326)        



## News
**[2021-10]**  Our dataset **V2X-Sim 1.0** is availale [here](https://ai4ce.github.io/V2X-Sim/).

**[2021-09]**  🔥 DiscoNet is accepted at **NeurIPS 2021**.

## Abstract
To promote better performance-bandwidth trade-off for multi-agent perception, we propose a novel distilled collaboration graph (DiscoGraph) to model trainable, pose-aware, and adaptive collaboration among agents. Our key novelties lie in two aspects. First, we propose a teacher-student framework to train DiscoGraph via knowledge distillation. The teacher model employs an early collaboration with holistic-view inputs; the student model is based on intermediate collaboration with single-view inputs. Our framework trains DiscoGraph by constraining post-collaboration feature maps in the student model to match the correspondences in the teacher model. Second, we propose a matrix-valued edge weight in DiscoGraph. In such a matrix, each element reflects the inter-agent attention at a specific spatial region, allowing an agent to adaptively highlight the informative regions. During inference, we only need to use the student model named as the distilled collaboration network (DiscoNet). Attributed to the teacher-student framework, multiple agents with the shared DiscoNet could collaboratively approach the performance of a hypothetical teacher model with a holistic view. Our approach is validated on V2X-Sim 1.0, a large-scale multi-agent perception dataset that we synthesized using CARLA and SUMO co-simulation. Our quantitative and qualitative experiments in multi-agent 3D object detection show that DiscoNet could not only achieve a better performance-bandwidth trade-off than the state-of-the-art collaborative perception methods, but also bring more straightforward design rationale. Our code is available on https://github.com/ai4ce/DiscoNet.

## Installation
For white-box attacks, we use point-based [PointRCNN](https://github.com/sshaoshuai/PointRCNN) as the target detector.  
```point_rcnn.py``` ```rcnn_net.py``` ```rpn.py``` in ```PointRCNN/lib/net``` were modified for introducing attacks.   
```kitti_dataset.py``` ```kitti_rcnn_dataset.py```  in ```PointRCNN/lib/datasets``` were modified for loading our customized nusc_kitti dataset.   
  
The rest code of PointRCNN is left untouched.
### Requirements
* Linux (tested on Ubuntu 20.04)
* Python 3.7
* PyTorch 1.8.0
* CUDA 11.2

### Create Anaconda Environment
```bash
conda env create -f disco.yaml
conda activate disco
```

## Dataset Preparation
Please download the official [nuscenes dataset](https://www.nuscenes.org/nuscenes)(v1.0-trainval)

Use ```nusc_to_kitti.py``` to generate the dataset.

```bash
python nusc_to_kitti.py nuscenes_gt_to_kitti [--dataroot "Your nuscenes dataroot"]
```

It will generate the dataset in the structure as follows.
```
FLAT
├── dataset
│   ├── nusc_kitti
│   │   ├──val_1000
│   │   │   ├──image_2
│   │   │   ├──ImageSets
│   │   │   ├──label_2
│   │   │   ├──pose
│   │   │   ├──velodyne
```

**NOTICE**: This script converts the first 1000(of 6019 in total) samples from orginal validation split of v1.0-trainval at default. You can use all of the nuscenes samples, and shuffle option is also provided.

## Training Commands
```bash
python train_codet.py [--data PATH_TO_DATA] [--bound BOUND] [--com COM] [--data PATH_TO_DATA]
               [--batch BATCH] [--nepoch NEPOCH] [--lr LEARNING_RATE] 
               [--kd_flag KD_FLAG] [--resume_teacher PATH_TO_TRACHER_MODEL]
```

```
--bound BOUND       
                    Input data to the collaborative perception model. Options: "lowerbound" for 
                    no-collaboration or intermediate-collaboration, "upperbound" for early collaboration.
--com COM   
                    Intermediate collaboration strategy. Options: "disco" for our DiscoNet,
                    "v2v/when2com//sum/mean/max/cat/agent" for other methods, '' for early or no collaboration.
--data PATH_TO_DATA         
                    Set as YOUR_PATH_TO_DATASET/V2X-Sim-1.0-trainval/train
--kd_flag FLAG
                    Whether to use knowledge distillation. 1 for true and 0 for false.
--resume_teacher PATH_TO_TRACHER_MODEL 
                    The pretrained early-collaboration-based teacher model.

```
All the experiments were performed at the [pretrained model](checkpoint_epoch_70.pth) of PointRCNN as provided.

Detection and evaluation results will be save in 
```bash
output/{SPLIT}/{ATTACK_TYPE}/FLAT_{STAGE}_{TASK}_{NB_ITER}_{ITER_EPS}_{ITER_EPS2}
```

## Acknowledgment  
```flat.py``` is modified from the evaluation code of [PointRCNN](https://github.com/sshaoshuai/PointRCNN), for implementing attacks.  
```evaluate.py``` is  borrowed from evaluation code from [Train in Germany, Test in The USA: Making 3D Object Detectors Generalize](https://github.com/cxy1997/3D_adapt_auto_driving), utilizing distance-based difficulty metrics.  
```nusc_to_kitti.py``` is  modified from official [nuscenes-devkit script](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/scripts/export_kitti.py) to generate kitti-format nuscenes dataset with ego pose for interpolation.  
* [PointRCNN](https://github.com/sshaoshuai/PointRCNN)
* [3D_adapt_auto_driving](https://github.com/cxy1997/3D_adapt_auto_driving)
* [nusSenes-devkit](https://github.com/nutonomy/nuscenes-devkit)

This project is not possible without these great codebases.

## Citation
If you find V2X-Sim 1.0 or DiscoNet useful in your research, please cite:
```
@InProceedings{Li_2021_NeurIPS,
    title = {Learning Distilled Collaboration Graph for Multi-Agent Perception},
    author = {Li, Yiming and Ren, Shunli and Wu, Pengxiang and Chen, Siheng and Feng, Chen and Zhang, Wenjun},
    booktitle = {Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021)},
    year = {2021}
}
```
