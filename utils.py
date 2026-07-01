import torch

def class_balanced_cross_entropy_loss(pred, label):
    """
    pred:  [N,1,H,W]  概率值 (0~1)
    label: [N,1,H,W]  二值标注 (0或1)
    """
    eps = 1e-7
    pred = torch.clamp(pred, eps, 1 - eps)
    label = (label > 0.5).float()

    num_positive = torch.sum(label).item()
    num_negative = torch.sum(1 - label).item()
    total = num_positive + num_negative

    if num_positive == 0:
        return -torch.mean(torch.log(1 - pred))

    w_pos = num_negative / total
    w_neg = num_positive / total

    loss = -w_pos * label * torch.log(pred) - w_neg * (1 - label) * torch.log(1 - pred)
    return loss.mean()