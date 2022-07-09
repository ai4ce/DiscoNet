# DiscoNet: Learning Distilled Collaboration Graph for Multi-Agent Perception [NeurIPS 2021]

[Yiming Li](https://scholar.google.com/citations?user=i_aajNoAAAAJ), [Shunli Ren](https://github.com/ShunliRen), [Pengxiang Wu](https://scholar.google.com/citations?user=MXLs7GcAAAAJ&hl=en), [Siheng Chen](https://scholar.google.com/citations?user=W_Q33RMAAAAJ&hl=en), [Chen Feng](https://scholar.google.com/citations?user=YeG8ZM0AAAAJ), [Wenjun Zhang](https://www.researchgate.net/profile/Wenjun-Zhang-29)

**"Learn a digraph with matrix-valued edge weight for multi-agent perception."**

<p align="center"><img src='img.png' align="center" height="400px"> </p>
  
  
## News
**[2022-07]**  We updated the codebase to [coperception library](https://github.com/coperception/coperception) and dataset to [V2X-Sim 2.0](https://ai4ce.github.io/V2X-Sim). New tasks including segmentation and tracking are included.

**[2021-11]**  Our paper is available on [arxiv](https://arxiv.org/pdf/2111.00643.pdf).

**[2021-10]**  Our dataset **V2X-Sim 1.0** is available [here](https://ai4ce.github.io/V2X-Sim/).

**[2021-09]**  🔥 DiscoNet is accepted at **NeurIPS 2021**.

## Abstract
To promote better performance-bandwidth trade-off for multi-agent perception, we propose a novel distilled collaboration graph (DiscoGraph) to model trainable, pose-aware, and adaptive collaboration among agents. Our key novelties lie in two aspects. First, we propose a teacher-student framework to train DiscoGraph via knowledge distillation. The teacher model employs an early collaboration with holistic-view inputs; the student model is based on intermediate collaboration with single-view inputs. Our framework trains DiscoGraph by constraining post-collaboration feature maps in the student model to match the correspondences in the teacher model. Second, we propose a matrix-valued edge weight in DiscoGraph. In such a matrix, each element reflects the inter-agent attention at a specific spatial region, allowing an agent to adaptively highlight the informative regions. During inference, we only need to use the student model named as the distilled collaboration network (DiscoNet). Attributed to the teacher-student framework, multiple agents with the shared DiscoNet could collaboratively approach the performance of a hypothetical teacher model with a holistic view. Our approach is validated on V2X-Sim 1.0, a large-scale multi-agent perception dataset that we synthesized using CARLA and SUMO co-simulation. Our quantitative and qualitative experiments in multi-agent 3D object detection show that DiscoNet could not only achieve a better performance-bandwidth trade-off than the state-of-the-art collaborative perception methods, but also bring more straightforward design rationale. Our code is available on https://github.com/ai4ce/DiscoNet.

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

@InProceedings{Li_2021_ICCVW,
    title = {V2X-Sim: A Virtual Collaborative Perception Dataset for Autonomous Driving},
    author = {Li, Yiming and An, Ziyan and Wang, Zixun and Zhong, Yiqi and Chen, Siheng and Feng, Chen},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
    year = {2021}
}
```
