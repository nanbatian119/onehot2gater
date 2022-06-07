from torch import nn
import numpy as np
from torch.functional import F
import torch
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.rel_pos_bias = nn.Linear(64,12,bias=False)

    def forward(self,rel_pos,hidden_states=None):
        rel_pos = F.one_hot(rel_pos, num_classes=64).type_as(hidden_states)
        rel_pos = self.rel_pos_bias(rel_pos)#.permute(0, 3, 1, 2)

        return rel_pos


