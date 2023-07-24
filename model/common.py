from turtle import forward
from typing import List
import torch.nn as nn

class SwishLayerNorm(nn.Module):
    def __init__(self, dim):
        super(SwishLayerNorm, self).__init__()
        self.norm = nn.Sequential(nn.LayerNorm(dim), nn.Sigmoid())
    
    def forward(self, input):
        return input * self.norm(input)

class MLP(nn.Module):
    def __init__(self, dims: List[int], add_bias=True, act="gelu", apply_layernorm=False, elemwise_affine=False):
        super().__init__()
        self._activation = self._get_activation(act)
        self._apply_layernorm = apply_layernorm
        self._elemwise_affine = elemwise_affine
        self._add_bias = add_bias
        self._model = self._create_model(dims)

    def _create_model(self, dims):
        layers = nn.ModuleList()
        for i in range(1, len(dims)):
            layer = nn.Linear(dims[i-1], dims[i]) if self._add_bias else nn.Linear(dims[i-1], dims[i], bias=False)
            layers.append(layer)

            if i < len(dims) - 1:
                if self._apply_layernorm:
                    layers.append(nn.LayerNorm(dims[i], elementwise_affine=self._elemwise_affine))

                layers.append(self._activation)
        
        return nn.Sequential(*layers)

    def _get_activation(self, act):
        if act == 'gelu':
            return nn.GELU()
        elif act == 'relu':
            return nn.ReLU()
        elif act == 'mish':
            return nn.Mish()
        elif act == 'tanh':
            return nn.Tanh()
        else:
            raise NotImplementedError


    def forward(self, input):
        return self._model(input)