from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.common import MLP

class NeighborAveragingSetRepresentationModel(nn.Module):
    def __init__(self, user_embed_dim: int, item_embed_dim: int, set_embedding_dim: int, hidden_dims: List[int], act='gelu'):
        super(NeighborAveragingSetRepresentationModel, self).__init__()
        self.dims = [user_embed_dim + item_embed_dim] + hidden_dims + [set_embedding_dim]
        self.mlp = MLP(self.dims, apply_layernorm=True, act=act, elemwise_affine=True)

    def forward(self, user_embeddings: torch.Tensor, neighbor_item_embeddings: torch.Tensor) -> torch.Tensor:
        """Outputs set representation given user_embeddings and interacted item embeddings

        Args:
            user_embeddings [B, d_u]: Embeddings of users in batch
            neighbor_item_embeddings [B, n, d_i]: Embeddings of neighboring items each user has interacted with 
        """
        neighbor_agg_embeds = torch.mean(neighbor_item_embeddings, dim=1) # Of shape [B, d_i]
        combined_embeds = torch.cat([user_embeddings, neighbor_agg_embeds], dim=1) # Of shape [B, d_u + d_i]
        return self.mlp(combined_embeds)


class MultiHeadAttentionSetRepresentationModel(nn.Module):
    def __init__(self, user_embed_dim: int, item_embed_dim: int, projection_dim: int, set_embedding_dim: int, hidden_dims: List[int], num_heads: int=1, act='gelu'):
        super(MultiHeadAttentionSetRepresentationModel, self).__init__()
        self.d_u = user_embed_dim
        self.d_i = item_embed_dim
        self.num_heads = num_heads
        self.projection_dim = projection_dim

        # Multihead attention parameters
        self.Q_w = nn.Linear(user_embed_dim, num_heads * projection_dim, bias=False)
        self.K_w = nn.Linear(item_embed_dim, num_heads * projection_dim, bias=False)
        self.mha_fc = nn.Linear(num_heads * item_embed_dim, num_heads * item_embed_dim)

        # User-Neighborhood combination parameters
        self.dims = [user_embed_dim + (num_heads * item_embed_dim)] + hidden_dims + [set_embedding_dim]
        self.mlp = MLP(self.dims, apply_layernorm=True, elemwise_affine=True, act=act)
    
    def forward(self, user_embeddings: torch.Tensor, neighbor_item_embeddings: torch.Tensor) -> torch.Tensor:
        """Outputs set representation given user_embeddings and interacted item embeddings

        Args:
            user_embeddings [B, d_u]: Embeddings of users in batch
            neighbor_item_embeddings [B, n, d_i]: Embeddings of neighboring items each user has interacted with 
        """
        batch_size = user_embeddings.shape[0]

        # Multihead attention to get neighborhood aggregate representation
        
        Q = self.Q_w(user_embeddings)
        K = self.K_w(neighbor_item_embeddings)

        Q = Q.view(batch_size, self.num_heads, 1, self.projection_dim) # Of shape [B, num_heads, 1, projection_dim]
        K = K.view(batch_size, self.num_heads, -1, self.projection_dim) # Of shape [B, num_heads, num_neighbors, projection_dim]
        V = neighbor_item_embeddings # Of shape [B, num_neighbors, item_embed_dim]

        weights = torch.squeeze(torch.matmul(Q, K.transpose(2, 3)), dim=2) / (self.projection_dim ** 0.5) # Of shape [B, num_heads, num_neighbors]
        weights = F.softmax(weights, dim=-1)

        neighbor_agg_embeds = torch.matmul(weights, V).view(batch_size, -1) # Of shape [B, num_heads * item_embed_dim]
        neighbor_agg_embeds = self.mha_fc(neighbor_agg_embeds)


        # Compute set embeddings

        combined_embeds = torch.cat([user_embeddings, neighbor_agg_embeds], dim=1)
        return self.mlp(combined_embeds)


class RecurrentSetRepresentationModel(nn.Module):
    def __init__(self, seq_len: int,
                       user_embed_dim: int, 
                       item_embed_dim: int, 
                       hidden_size: int, 
                       num_layers: int, 
                       set_embedding_dim: int, hidden_dims: List[int], act='gelu', mode='gru'):
        super(RecurrentSetRepresentationModel, self).__init__()
        self.seq_len = seq_len
        self.d_u = user_embed_dim
        self.d_i = item_embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode = mode

        if mode == 'gru':
            self.recurr_module = nn.GRU(item_embed_dim, hidden_size, num_layers, batch_first=True)
        else:
            self.recurr_module = nn.LSTM(item_embed_dim, hidden_size, num_layers, batch_first=True)


        # User-Neighborhood combination parameters
        self.dims = [(seq_len * hidden_size) + user_embed_dim] + hidden_dims + [set_embedding_dim]
        self.mlp = MLP(self.dims, apply_layernorm=True, elemwise_affine=True, act=act)
    
    def forward(self, user_embeddings: torch.Tensor, neighbor_item_embeddings: torch.Tensor) -> torch.Tensor:
        """Outputs set representation given user_embeddings and interacted item embeddings

        Args:
            user_embeddings [B, d_u]: Embeddings of users in batch
            neighbor_item_embeddings [B, seq_len, d_i]: Embeddings of neighboring items each user has interacted with 
        """
        batch_size = user_embeddings.shape[0]
        hidden_state = torch.randn(self.num_layers, batch_size, self.hidden_size, device=user_embeddings.device)
        if self.mode == 'gru':
            inital_state = hidden_state
        else:
            cell_state = torch.randn(self.num_layers, batch_size, self.hidden_size, device=user_embeddings.device)
            inital_state = (hidden_state, cell_state)
        
        recurr_output, _ = self.recurr_module(neighbor_item_embeddings, inital_state) # [B, num_layers, 2 * hidden_size]
        recurr_output = recurr_output.reshape(batch_size, -1)
        
        # Compute set embeddings
        combined_embeds = torch.cat([user_embeddings, recurr_output], dim=1)
        return self.mlp(combined_embeds)