import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from dataset import HEDDataset
from model import HED
from utils import class_balanced_cross_entropy_loss


def train():
    cfg = Config()
    cfg.create_dirs()

    # 加载完整训练集（通过 train_pair.lst）
    full_train_dataset = HEDDataset(cfg.data_root,
                                    split='train',
                                    use_all_scales=cfg.use_all_scales,
                                    dedup=cfg.dedup_images
                                    )

    # 划分训练集和验证集
    val_size = int(len(full_train_dataset) * cfg.val_ratio)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    num_workers = min(os.cpu_count() or 4, 4)
    persistent = num_workers > 0
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False, persistent_workers=persistent)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False, persistent_workers=persistent)

    model = HED().to(cfg.device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    writer = SummaryWriter(cfg.log_dir)

    best_val_loss = float('inf')

    for epoch in range(cfg.num_epochs):
        # ----- 训练阶段 -----
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{cfg.num_epochs} [Train]')
        for images, edges in pbar:
            images, edges = images.to(cfg.device), edges.to(cfg.device)

            optimizer.zero_grad()
            s1, s2, s3, s4, s5, fuse = model(images)

            loss1 = class_balanced_cross_entropy_loss(s1, edges)
            loss2 = class_balanced_cross_entropy_loss(s2, edges)
            loss3 = class_balanced_cross_entropy_loss(s3, edges)
            loss4 = class_balanced_cross_entropy_loss(s4, edges)
            loss5 = class_balanced_cross_entropy_loss(s5, edges)
            loss_fuse = class_balanced_cross_entropy_loss(fuse, edges)

            total_loss = (cfg.loss_weights[0] * loss1 + cfg.loss_weights[1] * loss2 +
                          cfg.loss_weights[2] * loss3 + cfg.loss_weights[3] * loss4 +
                          cfg.loss_weights[4] * loss5 + cfg.loss_weights[5] * loss_fuse)

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()
            pbar.set_postfix({'loss': total_loss.item()})

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # ----- 验证阶段 -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, edges in tqdm(val_loader, desc=f'Epoch {epoch + 1} [Val]'):
                images, edges = images.to(cfg.device), edges.to(cfg.device)
                _, _, _, _, _, fuse = model(images)
                loss = class_balanced_cross_entropy_loss(fuse, edges)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)

        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, cfg.model_name))
            print(f'Best model saved (val loss: {best_val_loss:.4f})')

    writer.close()


if __name__ == '__main__':
    train()