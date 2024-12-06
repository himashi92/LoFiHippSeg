# LoFiHippSeg: Bilateral Hippocampi Segmentation in Low Field MRIs Using Mutual Feature Learning via Dual-Views

First runner-up entry for segmentation Task in Low field pediatric brain magnetic resonance Image Segmentation and quality Assurance - LISA Challenge 2024. [Link to Challenge](https://www.synapse.org/Synapse:syn55249552/wiki/)

## Abstract
Accurate hippocampus segmentation in brain MRI is critical for studying cognitive and memory functions and diagnosing neurodevelopmental disorders. While high-field MRIs provide detailed imaging, low-field MRIs are more accessible and cost-effective, which eliminates the need for sedation in children, though they often suffer from lower image quality. In this paper, we present a novel deep-learning approach for the automatic segmentation of bilateral hippocampi in low-field MRIs. Extending recent advancements in infant brain segmentation to underserved communities through the use of low-field MRIs ensures broader access to essential diagnostic tools, thereby supporting better healthcare outcomes for all children. 
Inspired by our previous work, Co-BioNet, the proposed model employs a dual-view structure to enable mutual feature learning via high-frequency masking, enhancing segmentation accuracy by leveraging complementary information from different perspectives. Extensive experiments demonstrate that our method provides reliable segmentation outcomes for hippocampal analysis in low-resource settings. 

## Link to full paper:
Paper Link Pre-print : [Link](http://arxiv.org/abs/2410.17502)

## System requirements
Under this section, we provide details on the environmental setup and dependencies required to train/test the LoFiHippSeg model.
This software was originally designed and run on a system running Ubuntu.
<br>
All the experiments are conducted on Ubuntu 20.04 Focal version with Python 3.8.
<br>
To train LoFiHippSeg with the given settings, the system requires 1 GPU with at least 80GB. All the experiments are conducted on Nvidia A100 GPUs.
(Not required any non-standard hardware)
<br>
To test the model's performance on unseen test data, the system requires a GPU with at least 24 GB.

### Create a virtual environment

- Download anaconda:
 	   wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh

- Install anaconda:
   bash Anaconda3-2022.10-Linux-x86_64.sh

```bash 
conda create -n your_env_name python=3.8
conda activate your_env_name
```

### Installation guide 

- Install torch : 
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```

- Install other dependencies :
```bash 
pip install -r requirements.txt
```

### Typical Install Time 
This depends on the internet connection speed. It would take around 15-30 minutes to create environment and install all the dependencies required.

## Dataset Preparation
The experiments are conducted on LISA 2024 Challenge dataset.

## Pretrained Model
The pretrained models can be downloaded from here: https://drive.google.com/drive/folders/1PajSZf3T6xC8laeensI3gFcNX0OekqIH?usp=sharing

## Train Model

```bash
CUDA_VISIBLE_DEVICES=0 python train.py &> experiment.out &
```

## Test Model

```bash
cd code
CUDA_VISIBLE_DEVICES=0 python validation.py  &> experiment_evaluation.out &

```
