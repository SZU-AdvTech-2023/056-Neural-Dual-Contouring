### 环境配置
1. 使用环境：Ubuntu 21.04, 硬件为NVIDIA TITAN Xp
2. 安装cuda:https://developer.nvidia.com/cuda-downloads
3. 安装anaconda，创建虚拟环境，安装相关依赖
```
conda create -n implicit python=3.7
pip install -r requirements.txt
cd  models/ops/
sh make.sh
cd ...
```
### 数据集准备
点击这个链接下载 [link](https://drive.google.com/file/d/1BL58xl2U8H96YBkkB7WjDmtznuEj6PbG/view?usp=sharing). 并把数据解压到```./data```文件夹

### 模型训练

```
CUDA_VISIIBLE_DEVICE=0,1,2,3 python train_gifs_former.py
```

### 测试生成结果

```
cd s3d_floorplan_eval
python evaluate_solution.py
```
