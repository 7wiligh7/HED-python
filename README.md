# HED-python
python复现HED算法，通过5层卷积实现图片边缘检测

原算法链接https://github.com/s9xie/hed

数据集下载链接：https://vcl.ucsd.edu/hed/HED-BSDS.tar

全卷积VGG模型下载链接：https://vcl.ucsd.edu/hed/5stage-vgg.caffemodel

边缘检测算法——整体嵌套边缘检测（HED），它通过深度学习模型实现图像对图像的预测，该模型利用全卷积神经网络和深度监督网络。HED会自动学习丰富的层级表征（通过对侧面响应的深度监督指导），这些表示对于解决边缘和物体边界检测中的复杂歧义至关重要

这是我使用agent实现的hed算法的python语言复现

dataset.py：下载BSDS的数据集

config.py：配置文件，包含文件结构，保存位置，数据清洗（BSDS中存在大量的重复裁切图片，大大增加了耗时）

use_all_scales = False  #28800->9600

dedup_images = True  #9600->300

num_epochs：训练次数

learning_rate：学习率

model.py：模型文件

test.py：验证测试
