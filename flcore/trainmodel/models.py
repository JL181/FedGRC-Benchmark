import torch.nn as nn
class BaseHeadSplit(nn.Module):

    def __init__(self, base, head):
        super().__init__()
        self.base = base
        self.head = head

    def forward(self, x):
        return self.head(self.base(x))
