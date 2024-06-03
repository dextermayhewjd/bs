# 5.31 (1)  安装部分  installation
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

#mmcv 2.1.0
#pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html

#install mmaction2
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .

#install timm
pip install timm

```
```

```

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
torch 2.1.2 requires fsspec, which is not installed.
```
# Demo to prove the it works
4. verify the 
Verify the installation
To verify whether MMAction2 is installed correctly, we provide some sample codes to run an inference demo.

Step 1. Download the config and checkpoint files.
```ubuntu terminal
mim download mmaction2 --config tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb --dest .
```
Step 2. Verify the inference demo.

Option (a). If you install mmaction2 from source, you can run the following command:

The demo.mp4 and label_map_k400.txt are both from Kinetics-400
```ubuntu terminal
# The demo.mp4 and label_map_k400.txt are both from Kinetics-400
python demo/demo.py tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb.py \
    tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth \
    demo/demo.mp4 tools/data/kinetics/label_map_k400.txt
```
You will see the top-5 labels with corresponding scores in your terminal.


# Inference
```
python demo/demo_inferencer.py  demo/demo.mp4 \
    --rec tsn --print-result \
    --label-file tools/data/kinetics/label_map_k400.txt
# demo/demo.mp4 \ path 运行的视频路径
# -- rec tsn 要跑的algorithm
# --print-result 打印结果
# --label-file tools/data/kinetics/label_map_k400.txt # label file for dataset
```

You should be able to see a pop-up video and the inference result printed out in the console.  
If you are running MMAction2 on a server without a GUI or via an SSH tunnel with X11 forwarding disabled, you may not see the pop-up window.  
You should Install the VCXSRV WINDOWS X-SERVER https://sourceforge.net/projects/vcxsrv/files/latest/download  
Running the  LINUX DESKTOP and check the sdl2 if you on using ubuntu on Windows
```
#In an Ubuntu terminal, run 
export DISPLAY=$(grep -m 1 nameserver /etc/resolv.conf | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1
startlxde
```