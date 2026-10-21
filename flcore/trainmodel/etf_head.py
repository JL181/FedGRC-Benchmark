import math
import numpy as np
import torch
import torch.nn as nn

class ETFHead(nn.Module):

    def __init__(self, backbone_out_dim, num_classes):
        super().__init__()
        self.linear_proto = nn.Linear(backbone_out_dim, num_classes)
        self.scaling_train = nn.Parameter(torch.tensor(1.0))
        M = self._build_etf(num_classes, num_classes)
        self.register_buffer('etf_proto', M)

    @staticmethod
    def _build_etf(feat_in, num_classes):
        a = np.random.random(size=(feat_in, num_classes))
        (P, _) = np.linalg.qr(a)
        P = torch.tensor(P).float()
        I = torch.eye(num_classes)
        one = torch.ones(num_classes, num_classes)
        M = math.sqrt(num_classes / (num_classes - 1)) * torch.matmul(P, I - 1.0 / num_classes * one)
        return M

    def forward(self, x):
        projected = self.linear_proto(x)
        norm = torch.norm(projected, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        normalized = projected / norm
        logits = self.scaling_train * torch.matmul(normalized, self.etf_proto)
        return logits
