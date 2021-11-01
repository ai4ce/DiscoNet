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

## Run FLAT on Evaluation
```bash
python flat.py [--stage STAGE] [--nb_iter NB_ITER]
               [--task TASK] [--attack_type ATTACK_TYPE] 
               [--iter_eps ITER_EPS] [--iter_eps2 ITER_EPS2] [--poly]
```

```
--split SPLIT       
                    The data split for evaluation
--stage STAGE       
                    Attack stage of Point RCNN. Options: "1" for RPN
                    stage, "2" for RCNN stage
--nb_iter NB_ITER   
                    Number of attack iterations in PGD
--task TASK         
                    Task of attacking. Options: "cls" for classification,
                    "reg" for regression
--attack_type ATTACK_TYPE
                    Specify attack type. Options: "all", "translation",
                    "rotation"
--iter_eps ITER_EPS 
                    Primary PGD attack step size for each iteration, in
                    translation only/rotation only attacks, this parameter
                    is used.
--iter_eps2 ITER_EPS2
                    Secondary PGD attack step size for each iteration,
                    only effective when attack_type is "all" and poly mode
                    is disabled.
--poly              
                    Polynomial trajectory perturbation option. Notice: if
                    true, attack_type will be fixed(translation)
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
