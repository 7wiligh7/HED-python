import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from model import HED
from config import Config

def main():
    cfg = Config()
    device = cfg.device

    # 1. 加载模型
    model_path = os.path.join(cfg.checkpoint_dir, cfg.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}，请先训练模型。")
    model = HED().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型加载成功: {model_path}")

    # 2. 读取 test.lst
    test_lst_path = os.path.join(cfg.data_root, 'test.lst')
    if not os.path.exists(test_lst_path):
        raise FileNotFoundError(f"test.lst 不存在: {test_lst_path}")
    with open(test_lst_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # test.lst 每行格式：图像相对路径（可能包含空格？建议按行读取）
    test_image_paths = [line.split()[0] for line in lines]  # 只取第一个字段

    # 3. 创建输出目录
    output_dir = './results/test'
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    target_h, target_w = cfg.image_size
    # 4. 逐个处理
    for img_rel in tqdm(test_image_paths, desc="Processing test images"):
        img_full = os.path.join(cfg.data_root, img_rel)
        if not os.path.exists(img_full):
            print(f"警告: 图像不存在 {img_full}，跳过")
            continue

        img_original = cv2.imread(img_full)
        if img_original is None:
            print(f"警告: 无法读取图像 {img_full}，跳过")
            continue

        original_h, original_w = img_original.shape[:2]

        img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (target_w, target_h))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            s1, s2, s3, s4, s5, fuse = model(img_tensor)

        name_without_ext = os.path.splitext(os.path.basename(img_rel))[0]
        img_output_dir = os.path.join(output_dir, name_without_ext)
        os.makedirs(img_output_dir, exist_ok=True)

        outputs = {'s1': s1, 's2': s2, 's3': s3, 's4': s4, 's5': s5, 'fuse': fuse}
        for name, out in outputs.items():
            edge = out.squeeze().cpu().numpy()
            edge = cv2.resize(edge, (original_w, original_h))
            edge = (edge * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(img_output_dir, f'{name}.png'), edge)

    print(f"测试完成！结果保存在 {output_dir}")

if __name__ == '__main__':
    main()