# 5.31
1. remove all existing environment

conda actions below
```ubuntu terminal
# list all the existing env
conda env list

#删除每个环境。假设你有一个名为 openmmlab 的环境：
conda env remove -n openmmlab

# Deactivate Current Environment
conda deactivate

```

2. check compatibility for cuda and mvcc and pytorch
https://mmcv.readthedocs.io/zh-cn/latest/get_started/installation.html
3. install
for pytroch please vist https://pytorch.org/get-started/locally/#start-locally
or https://pytorch.org/get-started/previous-versions/

since the cuda version is 12.1 and based on the compatibility page it provided so the following installation command is below
```ubuntu terminal
conda create --name openmmlab python=3.9 -y
conda activate openmmlab

# install pytorch 
# CUDA 12.1 and pytorch 2.1.2
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

#install timm
pip install timm

```




