import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from config import Config

class HEDDataset(Dataset):
    def __init__(self, root_dir, split='train', use_all_scales=False, dedup=True):
        self.root_dir = root_dir
        self.split = split
        self.cfg = Config()
        self.use_all_scales = use_all_scales
        self.dedup = dedup

        if split == 'train':
            lst_file = os.path.join(root_dir, 'train_pair.lst')
        else:
            lst_file = os.path.join(root_dir, 'test.lst')

        with open(lst_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        raw_pairs = []
        for line in lines:
            parts = line.split()
            if split == 'train' and len(parts) >= 2:
                raw_pairs.append((parts[0], parts[1]))
            elif split == 'test':
                raw_pairs.append((parts[0], None))

        # === 新增：过滤 scale 版本（独立于 dedup）===
        if split == 'train' and not use_all_scales:
            original_len = len(raw_pairs)
            raw_pairs = [pair for pair in raw_pairs if 'scale' not in pair[0]]
            print(f"过滤 scale 后：从 {original_len} 条减少到 {len(raw_pairs)} 条")

        # === 去重（可选）===
        if dedup and split == 'train':
            seen = {}
            for img_rel, edge_rel in raw_pairs:
                base_name = os.path.basename(img_rel)
                if base_name not in seen:
                    seen[base_name] = (img_rel, edge_rel)
            filtered_pairs = list(seen.values())
            print(f"去重后：从 {len(raw_pairs)} 条减少到 {len(filtered_pairs)} 条")
            raw_pairs = filtered_pairs

        self.pairs = raw_pairs
        self.target_h, self.target_w = self.cfg.image_size
        print(f"{split} 集加载完成，共 {len(self.pairs)} 个样本")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_rel, edge_rel = self.pairs[idx]
        img_path = os.path.join(self.root_dir, img_rel)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"图像不存在: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.target_w, self.target_h))

        if self.split == 'train' and edge_rel is not None:
            edge_path = os.path.join(self.root_dir, edge_rel)
            edge = cv2.imread(edge_path, cv2.IMREAD_GRAYSCALE)
            if edge is None:
                raise FileNotFoundError(f"边缘图不存在: {edge_path}")
            edge = cv2.resize(edge, (self.target_w, self.target_h))
            edge = edge.astype(np.float32) / 255.0
            edge = (edge > 0.5).astype(np.float32)
        else:
            edge = np.zeros((self.target_h, self.target_w), dtype=np.float32)

        # 在线随机水平翻转（增强）
        if self.split == 'train' and np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            edge = cv2.flip(edge, 1)

        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        edge = torch.from_numpy(edge).float().unsqueeze(0)
        return image, edge