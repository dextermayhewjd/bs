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


# Training with pretrain model  
1. change the source of the dataset  
```python
data_root = 'data/kinetics400_tiny/train'
data_root_val = 'data/kinetics400_tiny/val'
ann_file_train = 'data/kinetics400_tiny/kinetics_tiny_train_video.txt'
ann_file_val = 'data/kinetics400_tiny/kinetics_tiny_val_video.txt'
```
2. change the batch size based on the size of the dataset
```python
train_dataloader['batch_size'] = 4
```  
3. change the hook to save the check points and keeep the latest checkpoint 
```python
default_hooks = dict(checkpoint=dict(interval=1, max_keep_ckpts=1))

```


4. 将最大 epoch 数设置为 10，并每 1 个 epoch验证模型
```python
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)

```  
Q: 此处的train_cfg 是哪里的呢  
A: inherit from the scheduler and the 
https://mmengine.readthedocs.io/en/latest/api/runner.html is the document for the EpochBasedTrainLoop used in the quick 20 mins run  


5. 根据 10 个 epoch调整学习率调度
```python
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=10,
        by_epoch=True,
        milestones=[4, 8],
        gamma=0.1)
]
```

# MMEngine
1. all parts  
Build a Model  
Build a Dataset and DataLoader
Build a Evaluation Metrics  
Build a Runner and Run the Task



To visualize the video after being sent through the pipeline
```terminal
python tools/visualizations/browse_dataset.py \
    configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb2.py \
    browse_out --mode pipeline
# ImportError: Please install moviepy to enable output file.  
pip install moviepy
```
# 6/5
1. 将数据分类 folder/ via and video 

# 6/29
1. 把训练模型的所有东西平凑齐全  
   数据集 问题不大  
   模型 现成的 如何改成需要的  -> 多任务
2. how to get the config done  
https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#config

# Config
1. 两种访问接口，即类似字典的接口 cfg['key'] 或者类似 Python 对象属性的接口 cfg.key。这两种接口都支持读写。
``` 
from mmengine.config import Config

cfg = Config.fromfile('learn_read_config.py')
print(cfg)

print(cfg.test_int)  
print(cfg.test_list)  
print(cfg.test_dict)  
cfg.test_int = 2  

print(cfg['test_int'])
print(cfg['test_list'])
print(cfg['test_dict'])
cfg['test_list'][1] = 3
print(cfg['test_list'])
```
```
1
[1, 2, 3]
{'key1': 'value1', 'key2': 0.1}
2
[1, 2, 3]
{'key1': 'value1', 'key2': 0.1}
[1, 3, 3]
```
2. 可以将配置与注册器结合起来使用，达到通过配置文件来控制模块构造的目的
```python
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
```
然后在算法库中可以通过如下代码构造优化器对象。
```python

from mmengine import Config, optim
from mmengine.registry import OPTIMIZERS

import torch.nn as nn

cfg = Config.fromfile('config_sgd.py')

model = nn.Conv2d(1, 1, 1)
cfg.optimizer.params = model.parameters()
optimizer = OPTIMIZERS.build(cfg.optimizer)
print(optimizer)
```

3. 继承机制  
1.为了解决这些问题，我们给配置文件增加了继承的机制，即一个配置文件 A 可以将另一个配置文件 B 作为自己的基础，直接继承了 B 中所有字段，而不必显式复制粘贴。                       
2.支持继承多个文件，将同时获得这多个文件中的所有字段，但是要求继承的多个文件中没有相同名称的字段，否则会报错。  
4. 修改继承字段  
有时候，我们继承一个配置文件之后，可能需要对其中个别字段进行修改，例如继承了 optimizer_cfg.py 之后，想将学习率从 0.02 修改为 0.01。  
这时候，只需要在新的配置文件中，重新定义一下需要修改的字段即可。注意由于 optimizer 这个字段是一个字典，我们只需要重新定义这个字典里面需修改的下级字段即可。这个规则也适用于增加一些下级字段。

resnet50_lr0.01.py：
```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(lr=0.01)
```
5. 删除字典中的 key (可能用不到暂时略过)  
有时候我们对于继承过来的字典类型字段，不仅仅是想修改其中某些 key，可能还需要删除其中的一些 key。这时候在重新定义这个字典时，需要指定 _delete_=True，表示将没有在新定义的字典中出现的 key 全部删除。 

resnet50_delete_key.py：
```python
_base_ = ['optimizer_cfg.py', 'runtime_cfg.py']
model = dict(type='ResNet', depth=50)
optimizer = dict(_delete_=True, type='SGD', lr=0.01)
```
这时候，optimizer 这个字典中就只有 type 和 lr 这两个 key，momentum 和 weight_decay 将不再被继承。

# Model
model is made of three thing

  1. model
  2. cls head
  3. data preprocessor

  编写 pipeline的时候需要考虑data preprocessor的输出格式
# Multiclass
1. 在video_dataset.py 中有关于如何制作数据集和为什么这么做的原因在multiclass的时候  
D:\mmac\mmaction2\mmaction\datasets  
eg.     
```python
def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split(self.delimiter)
            if self.multi_class:
                assert self.num_classes is not None
# 这里描述了多标签的情况可反推如何定义标签组
                filename, label = line_split[0], line_split[1:]
                
                label = list(map(int, label))
            # add fake label for inference datalist without label
            elif len(line_split) == 1:
                filename, label = line_split[0], -1
            else:
                filename, label = line_split
                label = int(label)
            if self.data_prefix['video'] is not None:
                filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))
        return data_list
```
2. timesformer没什么要改的 注意 num_class 要好好设置  
2.1有此处的BCELossWithLogits 输出的是概率数组  
2.2此处cls_head的in_channels通常与model中的emb_num相同
```python
cls_head=dict(
        type='TimeSformerHead',
        num_classes=20,
        in_channels=768,
        loss_cls= 'BCELossWithLogits',# beacause it is a multiclass classification and the output will be likelyhood or chances
        average_clips='prob'),# 存疑 在源代码中未看到
```

3. 找到如何写metric

4. 发现视频和标注 有问题 时间差了半秒 
 检察原因 
 首先是通过CVAT 可以查看每一帧的信息
 通过ffmpeg和pprobe查看帧

最后发现是因为有两个视频流 事变标注使用的是视频流0 而导出视频使用的是视频流1 而视频流1是只有1189帧 因为每秒20帧 符合差了半秒的

5. 需要做的 在视频分割出4秒视频前
   1. 首先需要检查视频流0和视频流1分别一共有多少帧
   2. 通过计算得知视频0标注的时间在视频1中是多少时间
   3. 因为需要统一视频大小以及视频数据集本身的一致性 所以一切开始时间小于视频difference时间的 开始的视频都去除

现在需要做的是 将视频首先 生成 一种格式方便查看 另一种方便机器理解

6.1.1 需要json file 以及video file 用于 算法  
6.1.2 需要 output file 用于存放视频文件  
6.1.3 需要 了解dataset在哪 

1.首先 paser需要 获得json file的路径
然后根据 json file 获得视频本身的路径 来生成算法 done
2. output file 的存放的路径应该怎样才方便检查 搬运和分类呢
3. dataset 是否需要分开多个 以供分批检查读写呢

/dog_video/Feb/1_31_unknown
# Data Directory Structure

```plaintext
dog_video/Feb/2_2_kt (base_dir)
        │
        ├── video(video_dir)
        │   ├── 020500.mp4  # Raw video files
        │   └── .....       # Additional video files
        │
        ├── via
        │   ├── 0205.json   # Annotations in JSON format
        │   ├── ....        # Json_file_path
        │   └── ....        # Additional annotation files
        │
        └── output(output_dir)
            ├──0205
            │   ├── 20240202kt_020500_start0.0_2 
            │   ├── .......
            │   └── # Processed output files
            ├──.......      # Processed output files time
            ├──.......
            └──dataset.txt  # Summary of datasets or additional information
```
