# losses/utils.py
import torch

def normalize_gradient_per_frame(g, clip_val=5.0, eps=1e-6):
    """
    g: [B, T] 或可 broadcast 至 [B, T]
    归一化 + 截断，避免除 0 与梯度爆炸
    """
    if g.dim() == 1:
        g = g.unsqueeze(0)
    norm = torch.linalg.norm(g, ord=1, dim=-1, keepdim=True)
    g = g / (norm + eps)
    return torch.clamp(g, -clip_val, clip_val)
