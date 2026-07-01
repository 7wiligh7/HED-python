import os
import cv2
import torch
import numpy as np
from model import HED
from config import Config

def predict(image_path, model_path, output_path):
    cfg = Config()
    device = cfg.device

    # 加载模型
    model = HED().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    original_h, original_w = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 预处理：resize 到训练尺寸，归一化
    image_resized = cv2.resize(image_rgb, (cfg.image_size[1], cfg.image_size[0]))  # (w,h)
    image_tensor = torch.from_numpy(image_resized).float().permute(2,0,1).unsqueeze(0) / 255.0
    image_tensor = image_tensor.to(device)

    # 推理
    with torch.no_grad():
        _, _, _, _, _, fuse = model(image_tensor)

    # 后处理
    edge_map = fuse.squeeze().cpu().numpy()
    edge_map = cv2.resize(edge_map, (original_w, original_h))
    edge_map = (edge_map * 255).astype(np.uint8)

    cv2.imwrite(output_path, edge_map)
    print(f"Edge map saved to {output_path}")

if __name__ == '__main__':
    cfg = Config()
    predict(cfg.input_image, os.path.join(cfg.checkpoint_dir, cfg.model_name), cfg.output_edge)