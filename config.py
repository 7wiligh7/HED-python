import torch
import os

class Config:
    # 数据集根目录
    data_root = r'./data/HED-BSDS'

    # 训练参数
    batch_size = 8
    learning_rate = 1e-4
    weight_decay = 0.0002
    num_epochs = 1

    # 图像尺寸（训练时 resize 到该尺寸）
    image_size = (480, 480)   # (height, width)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 损失函数权重（5个侧输出 + 融合输出）
    loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # 保存路径
    checkpoint_dir = './checkpoints'
    model_name = 'hed.pth'
    log_dir = './runs/hed_training'

    # 推理时输入图像路径（predict.py 中可修改）
    input_image = './test.jpg'
    output_edge = './results/edge.png'

    use_all_scales = False
    dedup_images = False

    val_ratio = 0.1
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(cls.log_dir), exist_ok=True)
        os.makedirs('./results', exist_ok=True)