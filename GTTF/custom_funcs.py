import numpy as np
import torch

from GTTF.framework import WalkForest, CompactAdj
from data.structs import InteractionGraph


def uniform_degree_prob(node, compact_adj: CompactAdj, *args):
    neighbors = compact_adj[node]
    probs = np.ones_like(neighbors)
    return probs / probs.sum()

def softmax_degree_prob(node, compact_adj: CompactAdj, temp: int):
    neighbors = compact_adj[node]
    neighbor_degrees = compact_adj.edge_lccsr.degrees[neighbors]
    inverse_degrees = 1 / neighbor_degrees
    inverse_degrees = inverse_degrees - inverse_degrees.max()
    probs = np.exp(inverse_degrees / temp)
    return probs / probs.sum()

def accumulate_one_hop_neighbors(walk_forest: WalkForest, edge_items: np.ndarray = None, filter: bool = False, igraph: InteractionGraph = None):
    starting_nodes = walk_forest[0].reshape(-1, 1)
    one_hop_neighbors = walk_forest[1].reshape(starting_nodes.shape[0], -1)

    # Replace the true edge items in the samples with another item from the samples, sampled randomly
    if filter:
        edge_items = edge_items.reshape(-1, 1)
        hitting_rows, hitting_cols = (one_hop_neighbors == edge_items).nonzero()
        probs = torch.as_tensor((one_hop_neighbors == edge_items) == False, dtype=torch.float)
        replacement_item_column_idxs = torch.multinomial(probs, 1).view(-1).numpy()[hitting_rows]
        one_hop_neighbors[hitting_rows, hitting_cols] = one_hop_neighbors[hitting_rows, replacement_item_column_idxs]

    one_hop_edges = np.stack([starting_nodes.repeat(one_hop_neighbors.shape[1], axis=1), one_hop_neighbors], axis=2).reshape(-1,2)
    return one_hop_edges
