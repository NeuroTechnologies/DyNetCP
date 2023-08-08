import torch
import torch.nn as nn

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.unsqueeze(self.dim)

def get_positional_encoding(d, t):
    d_ind = torch.arange(0, int(d/2), dtype=torch.float)
    t_ind = torch.arange(0, t, dtype=torch.float)
    omega = 1 / 10000**(2*d_ind/d)
    tmp = t_ind.unsqueeze(-1)*omega.unsqueeze(0)
    result = torch.stack([
        torch.sin(tmp),
        torch.cos(tmp)
    ], -1).reshape(t, d)
    return result