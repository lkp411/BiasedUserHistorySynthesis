from typing import Callable, Optional
import numpy as np
import torch

from GTTF.left_contiguous_csr import LeftContiguousCSR


class CompactAdj:
    def __init__(self, edge_lccsr: LeftContiguousCSR):
        self.edge_lccsr = edge_lccsr

    def __len__(self):
        return len(self.edge_lccsr)

    def __getitem__(self, item):
        return self.edge_lccsr[item]

    def sample_neighbors_uniform(self, batch, num_neighbors):
        neighbors = self.edge_lccsr.data[
          np.floor(
            np.random.uniform(size=(batch.size, num_neighbors)) * self.edge_lccsr.degrees[batch].reshape(-1,1)
          ).astype(np.int32) + self.edge_lccsr.indptr[batch].reshape(-1,1)
        ].reshape(*batch.shape, num_neighbors)
        return neighbors

    def create_multinomial_sampler(self, prob_for_node, start_item_id: int, temp: Optional[int], order=False):
        probs = []
        for i in range(0, start_item_id):
            probs.append(prob_for_node(i, self, temp))

        def sampler(batch, num_neighbors):
            flat_batch = batch.flatten()
            next_nodes = []
            for node in flat_batch:
                distribution = probs[node]
                replacement = num_neighbors > distribution.shape[0]
                sampled = torch.multinomial(torch.tensor(distribution), num_neighbors, replacement=replacement).numpy()
                nexts = self[node][sampled]
                if order:
                    ordered_timestaps = np.argsort(self.edge_lccsr.get_timestamps(node)[sampled])
                    nexts = nexts[ordered_timestaps]

                next_nodes.append(nexts)
            next_nodes = np.concatenate(next_nodes, axis=0) 
            return next_nodes.reshape(*batch.shape, num_neighbors)
        return sampler


class WalkForest:
    def __init__(self, batch, compact_adj: CompactAdj, fanouts, sampler: Callable):
        self.levels = [batch]
        for f in fanouts:
            previous_nodes = self.levels[-1]
            sampled_neighbors = sampler(previous_nodes, f)
            self.levels.append(sampled_neighbors)

    def __getitem__(self, item):
        return self.levels[item]

    def __len__(self):
        return len(self.levels)


def gttf(compact_adj: CompactAdj, bias_func=None, acc_func=None):
    def function_to_run(batch, fanouts, **kwargs):
        wf = WalkForest(batch, compact_adj, fanouts, sampler=bias_func)
        return wf if acc_func is None else acc_func(wf, **kwargs)
    return function_to_run



