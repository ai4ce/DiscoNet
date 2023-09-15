# DiscoNet: Learning Distilled Collaboration Graph for Multi-Agent Perception [NeurIPS 2021]

[Yiming Li](https://scholar.google.com/citations?user=i_aajNoAAAAJ), [Shunli Ren](https://github.com/ShunliRen), [Pengxiang Wu](https://scholar.google.com/citations?user=MXLs7GcAAAAJ&hl=en), [Siheng Chen](https://scholar.google.com/citations?user=W_Q33RMAAAAJ&hl=en), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ), [Wenjun Zhang](https://www.researchgate.net/profile/Wenjun-Zhang-29)

**"Learn a digraph with matrix-valued edge weight for multi-agent perception."**  

[![Documentation Status](https://readthedocs.org/projects/coperception/badge/?version=latest)](https://coperception.readthedocs.io/en/latest/?badge=latest)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
[![GitLab issues total](https://badgen.net/github/issues/ai4ce/DiscoNet)](https://gitlab.com/ai4ce/V2X-Sim/issues)
[![GitHub stars](https://img.shields.io/github/stars/ai4ce/DiscoNet.svg?style=social&label=Star&maxAge=2592000)](https://GitHub.com/ai4ce/V2X-Sim/stargazers/)

<p align="center"><img src='img.png' align="center" height="400px"> </p>

## News
**[2022-07]**  We updated the codebase to [coperception library](https://github.com/coperception/coperception) and dataset to [V2X-Sim 2.0](https://ai4ce.github.io/V2X-Sim). New tasks including segmentation and tracking are included.

**[2021-11]**  Our paper is available on [arxiv](https://arxiv.org/pdf/2111.00643.pdf).

**[2021-10]**  Our dataset **V2X-Sim 1.0** is available [here](https://ai4ce.github.io/V2X-Sim/).

**[2021-09]**  🔥 DiscoNet is accepted at **NeurIPS 2021**.

## Abstract
To promote better performance-bandwidth trade-off for multi-agent perception, we propose a novel distilled collaboration graph (DiscoGraph) to model trainable, pose-aware, and adaptive collaboration among agents. Our key novelties lie in two aspects. First, we propose a teacher-student framework to train DiscoGraph via knowledge distillation. The teacher model employs an early collaboration with holistic-view inputs; the student model is based on intermediate collaboration with single-view inputs. Our framework trains DiscoGraph by constraining post-collaboration feature maps in the student model to match the correspondences in the teacher model. Second, we propose a matrix-valued edge weight in DiscoGraph. In such a matrix, each element reflects the inter-agent attention at a specific spatial region, allowing an agent to adaptively highlight the informative regions. During inference, we only need to use the student model named as the distilled collaboration network (DiscoNet). Attributed to the teacher-student framework, multiple agents with the shared DiscoNet could collaboratively approach the performance of a hypothetical teacher model with a holistic view. Our approach is validated on V2X-Sim 1.0, a large-scale multi-agent perception dataset that we synthesized using CARLA and SUMO co-simulation. Our quantitative and qualitative experiments in multi-agent 3D object detection show that DiscoNet could not only achieve a better performance-bandwidth trade-off than the state-of-the-art collaborative perception methods, but also bring more straightforward design rationale. Our code is available on https://github.com/ai4ce/DiscoNet.

## Getting started
Please refer to the docs website of `coperception` for detailed documentations: https://coperception.readthedocs.io/en/latest/  
Installation:
- [Installation documentations](https://coperception.readthedocs.io/en/latest/getting_started/installation/)

Download dataset:
- Original dataset (you are going to parse this dataset yourself with `create_data.py` scripts for specific tasks): [Google Drive (US)](https://drive.google.com/drive/folders/1nVmY7g_kprOX-I0Bqsiz6-zdJM-UXFXa)  
- Parsed datasets for detection and segmentation tasks and model checkpoints: [Google Drive (US)](https://drive.google.com/drive/folders/1NMag-yZSflhNw4y22i8CHTX5l8KDXnNd?usp=sharing)   

How to create datasets & run tasks:
- [Detection](https://coperception.readthedocs.io/en/latest/tools/det/)
- [Segmentation](https://coperception.readthedocs.io/en/latest/tools/seg/)
- [Tracking](https://coperception.readthedocs.io/en/latest/tools/track/)

Example arguments to run DiscoNet:  
(assume dataset created properly according to the documentations)
- Detection / segmentation:  
    Inside coperception codebase:
    ```
    cd tools/det
    ```
    or
    ```
    cd tools/seg
    ```

    Training:
    ```bash
	python train_codet.py \
        --data  /path/to/training/dataset \
        --com disco \
        --log --batch 4 \
        --kd_flag 1 \
        --resume_teacher /path/to/teacher/checkpoint.pth \
        --auto_resume_path logs \
        --logpath logs \
        --nepoch 100 \
        -- rsu [0/1]
    ```

    Testing:
    ```bash
	python test_codet.py \
        --data /path/to/testing/dataset \
        --com disco \
        --resume /path/to/teacher/checkpoint.pth \
        --tracking \
        --logpath logs \
        --visualization 1 \
        --rsu 1
    ```

- Tracking:
    Inside coperception codebase:
    ```
    cd tools/track
    ```
    Run tracker:
    ```bash
    make sort \
        mode=disco/[no_rsu/with_rsu] \
        split [test/val] \
        from_agent 1 \
        to_agent 6 \
        det_logs_path /path/to/detection/checkpoints.pth 
    ```
    See performance:
    ```bash
    make eval
        mode=disco \
        rsu=[0/1] \
        from_agent=1 \
        to_agent=6 \
        split=test
    ```

## Acknowledgment  
This project is not possible without the following great codebases.
* [**MotionNet**](https://github.com/pxiangwu/MotionNet)
* [**mmdetection**](https://github.com/open-mmlab/mmdetection)
* [**nuSenes-devkit**](https://github.com/nutonomy/nuscenes-devkit)
* [**when2com**](https://github.com/GT-RIPL/MultiAgentPerception)
* [**V2X-Sim**](https://github.com/ai4ce/V2X-Sim)

## Citation
If you find V2X-Sim 1.0 or DiscoNet useful in your research, please cite our paper.
```
@InProceedings{Li_2021_NeurIPS,
    title = {Learning Distilled Collaboration Graph for Multi-Agent Perception},
    author = {Li, Yiming and Ren, Shunli and Wu, Pengxiang and Chen, Siheng and Feng, Chen and Zhang, Wenjun},
    booktitle = {Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021)},
    year = {2021}
}

@article{Li_2021_RAL,
    title = {V2X-Sim: Multi-Agent Collaborative Perception Dataset and Benchmark for Autonomous Driving},
    author = {Li, Yiming and Ma, Dekun and An, Ziyan and Wang, Zixun and Zhong, Yiqi and Chen, Siheng and Feng, Chen},
    journal = {IEEE Robotics and Automation Letters},
    year = {2022}
}
```
