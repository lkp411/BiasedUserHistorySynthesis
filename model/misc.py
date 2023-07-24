from typing import List
import torch
import torch.nn as nn
from model.common import MLP

class ItemEmbeddingModel(nn.Module):
    def __init__(self, item_feat_dim: int, hidden_dims: List[int], item_embed_dim: int, act='gelu'):
        super(ItemEmbeddingModel, self).__init__()
        self.dims = [item_feat_dim] + hidden_dims + [item_embed_dim]
        self.mlp = MLP(self.dims, apply_layernorm=True, act=act, elemwise_affine=True)
    
    def forward(self, input):
        return self.mlp(input)

class DotCompressScoringModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], act='gelu'):
        super(DotCompressScoringModel, self).__init__()
        self.dot_compress_weight = nn.Parameter(torch.empty(2, input_dim // 2))
        nn.init.xavier_normal_(self.dot_compress_weight)
        
        self.dot_compress_bias = nn.Parameter(torch.zeros(input_dim // 2))

        self.dims = [input_dim] + hidden_dims + [1]
        self.output_layer = MLP(self.dims, apply_layernorm=True, elemwise_affine=True)
    
    def forward(self, set_embeddings, item_embeddings):
        all_embeddings = torch.stack([set_embeddings, item_embeddings], dim=1)
        combined_representation = torch.matmul(all_embeddings, torch.matmul(all_embeddings.transpose(1, 2), self.dot_compress_weight) + self.dot_compress_bias).flatten(1)
        output = self.output_layer(combined_representation)
        return output
    
class DotProduct(nn.Module):
    def __init__(self):
        super(DotProduct, self).__init__()
    
    def forward(self, set_embeddings, item_embeddings):
        # set_embeddings = set_embeddings / torch.linalg.norm(set_embeddings, ord=2, dim=1, keepdim=True)
        # item_embeddings = item_embeddings / torch.linalg.norm(item_embeddings, ord=2, dim=1, keepdim=True)
        return torch.sum(set_embeddings * item_embeddings, dim=1)