import torch
from torch import nn


class BaseModel(nn.Module):

    def loss(self, inputs, labels):
        raise NotImplementedError()

    def predict(self, inputs, labels):
        raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(torch.load(path, map_location='cpu'))