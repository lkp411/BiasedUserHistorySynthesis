# Adapted from https://github.com/Elanmarkowitz/KGCompletionGNN

from collections import defaultdict
import array
import numpy as np
import tqdm

from GTTF.left_contiguous_csr import LeftContiguousCSR
from data.structs import InteractionGraph
from GTTF.framework import CompactAdj


def create_lccsr(num_entities, num_edges, edges, edge_timestamps, directed=False):
    # dictionary of edges
    edge_dict = defaultdict(lambda: array.array('i'))
    edge_timestamp_dict = defaultdict(lambda: array.array('d'))
    degrees = np.zeros((num_entities,), dtype=np.int32)
    indegrees = np.zeros((num_entities,), dtype=np.int32)
    outdegrees = np.zeros((num_entities,), dtype=np.int32)
    print("Building edge dict.")
    for h,t in tqdm.tqdm(edges):
        edge_dict[h].append(t)
        degrees[h] = degrees[h] + 1
        outdegrees[h] = outdegrees[h] + 1
        
        if edge_timestamps is not None:
            edge_timestamp_dict[h].append(edge_timestamps[(h, t)])

        if not directed:
            edge_dict[t].append(h)
            degrees[t] = degrees[t] + 1
            indegrees[t] = indegrees[t] + 1
            
            if edge_timestamps is not None:
                edge_timestamp_dict[t].append(edge_timestamps[(h, t)])


    print("Converting to np arrays.")
    total_num_edges = num_edges if directed else 2 * num_edges
    edge_csr_data = np.zeros((total_num_edges,), dtype=np.int32)
    edge_csr_timestamps = np.zeros((total_num_edges,), dtype=np.int64) if edge_timestamps is not None else None
    edge_csr_indptr = np.zeros((num_entities + 1,), dtype=np.int32)
    num_prev = 0
    for i in tqdm.tqdm(range(num_entities)):
        deg = degrees[i]
        edge_csr_indptr[i] = num_prev
        edge_csr_data[num_prev:num_prev + deg] = np.array(edge_dict[i], dtype=np.int32)
        if edge_timestamps is not None:
            edge_csr_timestamps[num_prev:num_prev + deg] = np.array(edge_timestamp_dict[i], dtype=np.int64)
        num_prev += degrees[i]
    edge_csr_indptr[-1] = num_prev

    edge_lccsr = LeftContiguousCSR(edge_csr_indptr, degrees, edge_csr_data, edge_csr_timestamps)
    return edge_lccsr, degrees, indegrees, outdegrees


def convert_to_compact_adj(graph: InteractionGraph):
    num_entities = len(graph.user_data) + len(graph.item_data)
    num_edges = len(graph.train_edges)
    edge_lccsr, degrees, indegrees, outdegrees = create_lccsr(num_entities, num_edges, graph.train_edges, graph.interaction_time_stamps)
    return CompactAdj(edge_lccsr)

