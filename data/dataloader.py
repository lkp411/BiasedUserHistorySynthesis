import random
import torch
import numpy as np
from torch.utils.data import Dataset

from data.structs import MovielensInteractionGraph, BookCrossingInteractionGraph
from timeit import default_timer as timer
from tqdm import tqdm
import os


#region MovieLens ---------------------------
class MovieLensDataset(Dataset):
    def __init__(self, igraph : MovielensInteractionGraph, mode='train') -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data
        self.item_data = igraph.item_data
        
        if mode == 'train':
            self.edges = igraph.train_edges
        elif mode == 'val':
            self.edges = igraph.validation_edges
        else:
            self.edges = igraph.test_edges

    
    def __len__(self):
        return self.edges.shape[0]
    
    def __getitem__(self, index):
        return self.edges[index]

class MovieLensCollator:
    def __init__(self, igraph: MovielensInteractionGraph, mode: str, num_neg_samples=1) -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data
        self.item_data = igraph.item_data
        self.adj_matrix = igraph.adj_matrix
        self.num_neg_samples = num_neg_samples
        self.mode = mode
        self.rng = np.random.Generator(np.random.PCG64(seed=0))

    def _generate_in_and_oob_negatives(self, positive_edges):
        item_start_id = len(self.user_data)
        pos_edges = np.array(positive_edges)

        negative_edges = []
        for i, (user_id, _) in enumerate(positive_edges):

            # Out of batch negative Sampling
            candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
            candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

            valid_samples = False
            while not valid_samples:
                neg_items = np.random.choice(candidate_item_probs.shape[0], (self.num_neg_samples,), p=candidate_item_probs)     
                for neg_item in neg_items:
                    if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                        valid_samples = False
                        break

                    valid_samples = True
            
            # In batch negative sampling
            in_batch_candidates = np.concatenate((pos_edges[:i], pos_edges[(i+1):]))[:, 1]
            idxs_to_delete = [idx for idx, candidate in enumerate(in_batch_candidates) if (user_id, candidate) in self.igraph.all_edges]
            valid_candidates = np.delete(in_batch_candidates, idxs_to_delete)
            in_batch_negs = self.rng.choice(valid_candidates, (self.num_neg_samples,), replace=False)

            for neg_item in neg_items:
                negative_edges.append([user_id, neg_item + item_start_id])

            for neg_item in in_batch_negs:
                negative_edges.append([user_id, neg_item])
        
        return negative_edges

    def _get_edges_to_score(self, edges):
        item_start_id = len(self.user_data)
        offsets = []
        edges_to_score = []
        for user_id, _ in edges:
            current_offset = len(edges_to_score)
            offsets.append(current_offset)
            negatives = np.argwhere(np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0).flatten() # The true validation and test items will be a part of this array
            edges_to_score += [[user_id, negative + item_start_id] for negative in negatives]
        
        return np.asarray(edges_to_score), np.asarray(offsets)
    
    def _fetch_data(self, edges):
        user_feats = [(user_id, self.user_data[user_id]['gender'], self.user_data[user_id]['age'], self.user_data[user_id]['occupation']) for user_id, _ in edges]
        item_feats = [(movie_id, self.item_data[movie_id]['date'], self.item_data[movie_id]['genres']) for _, movie_id in edges]
        item_title_embeddings = [self.item_data[movie_id]['title_embedding'] for _, movie_id in edges]

        return user_feats, item_feats, item_title_embeddings

    def __call__(self, positive_edges):
        if self.mode != 'train':
            true_edges = np.array(positive_edges)
            edges_to_score, offsets = self._get_edges_to_score(true_edges)
            return torch.as_tensor(true_edges, dtype=torch.int32), torch.as_tensor(edges_to_score, dtype=torch.int32), torch.as_tensor(offsets, dtype=torch.int32)

        negative_edges = self._generate_in_and_oob_negatives(positive_edges)
        edges = positive_edges + negative_edges
        user_feats, item_feats, item_title_embeddings = self._fetch_data(edges)

        user_feats_by_type = list(zip(*user_feats))
        user_ids = torch.as_tensor(user_feats_by_type[0], dtype=torch.int64)
        user_genders = torch.as_tensor(user_feats_by_type[1], dtype=torch.int64)
        user_ages = torch.as_tensor(user_feats_by_type[2], dtype=torch.int64)
        user_occs = torch.as_tensor(user_feats_by_type[3], dtype=torch.int64)

        item_feats_by_type = list(zip(*item_feats))
        movie_ids = torch.as_tensor(item_feats_by_type[0], dtype=torch.int64) - len(self.user_data)
        movie_dates = torch.as_tensor(item_feats_by_type[1], dtype=torch.int64)
        movie_genres = torch.as_tensor(np.asarray(item_feats_by_type[2]), dtype=torch.float32)
        movie_title_embeddings = torch.as_tensor(np.asarray(item_title_embeddings), dtype=torch.float32)

        return (user_ids, user_genders, user_ages, user_occs), (movie_ids, movie_dates, movie_genres, movie_title_embeddings)
        
class MovieLensInferenceItemsDataset(Dataset):
    def __init__(self, igraph : MovielensInteractionGraph) -> None:
        self.all_item_ids = sorted(list(igraph.item_data.keys()))

        self.item_reindexer = {}
        for item_id in self.all_item_ids:
            self.item_reindexer[item_id] = len(self.item_reindexer)
        
        self.reverse_item_indexer = {v : k for k, v in self.item_reindexer.items()}

    def __len__(self):
        return len(self.all_item_ids)
    
    def __getitem__(self, index):
        return self.all_item_ids[index]

class MovieLensItemsCollator:
    def __init__(self, igraph : MovielensInteractionGraph) -> None:
        self.igraph = igraph
        self.item_data = igraph.item_data
    
    def __call__(self, batch):
        item_feats = [(movie_id, self.item_data[movie_id]['date'], self.item_data[movie_id]['genres']) for movie_id in batch]
        item_title_embeddings = [self.item_data[movie_id]['title_embedding'] for movie_id in batch]

        item_feats_by_type = list(zip(*item_feats))
        movie_ids = torch.as_tensor(item_feats_by_type[0], dtype=torch.int64)
        zero_indexed_movie_ids = movie_ids - len(self.igraph.user_data)
        movie_dates = torch.as_tensor(item_feats_by_type[1], dtype=torch.int64)
        movie_genres = torch.as_tensor(np.asarray(item_feats_by_type[2]), dtype=torch.float32)
        movie_title_embeddings = torch.as_tensor(np.asarray(item_title_embeddings), dtype=torch.float32)

        return (movie_ids, zero_indexed_movie_ids, movie_dates, movie_genres, movie_title_embeddings)

class MovieLensInferenceUsersDataset(Dataset):
    def __init__(self, igraph : MovielensInteractionGraph) -> None:
        self.all_user_ids = sorted(list(igraph.user_data.keys()))
    
    def __len__(self):
        return len(self.all_user_ids)
    
    def __getitem__(self, index):
        return self.all_user_ids[index]

class MovieLensUsersCollator:
    def __init__(self, igraph : MovielensInteractionGraph) -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data

    def __call__(self, batch):
        user_feats = [(user_id, self.user_data[user_id]['gender'], self.user_data[user_id]['age'], self.user_data[user_id]['occupation']) for user_id in batch]
        user_feats_by_type = list(zip(*user_feats))
        user_ids = torch.as_tensor(user_feats_by_type[0], dtype=torch.int64)
        user_genders = torch.as_tensor(user_feats_by_type[1], dtype=torch.int64)
        user_ages = torch.as_tensor(user_feats_by_type[2], dtype=torch.int64)
        user_occs = torch.as_tensor(user_feats_by_type[3], dtype=torch.int64)

        return (user_ids, user_genders, user_ages, user_occs)
#endregion


#region BookCrossing -----------------------------
class BookCrossingDataset(Dataset):
    def __init__(self, igraph : BookCrossingInteractionGraph, mode='train') -> None:
        super().__init__()
        self.igraph = igraph
        self.user_data = igraph.user_data
        self.item_data = igraph.item_data
        
        if mode == 'train':
            self.edges = igraph.train_edges
        elif mode == 'val':
            self.edges = igraph.validation_edges
        else:
            self.edges = igraph.test_edges
    
    def __len__(self):
        return self.edges.shape[0]
    
    def __getitem__(self, index):
        return self.edges[index]

class BookCrossingCollator:
    def __init__(self, igraph: BookCrossingInteractionGraph, mode: str, num_neg_samples=1) -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data
        self.item_data = igraph.item_data
        self.adj_matrix = igraph.adj_matrix
        self.num_neg_samples = num_neg_samples
        self.mode = mode
        self.rng = np.random.Generator(np.random.PCG64(seed=0))

        if self.mode != 'train':
            self.user_negatives = self._compute_inference_negatives(num_negatives=40000)
    
    def _compute_inference_negatives(self, num_negatives):
        if os.path.exists('user_negatives.npy'):
            print('Loading user negatives from disk')
            with open('user_negatives.npy', 'rb') as f:
                user_negatives = np.load(f)
            
            return user_negatives

        else:
            print('Computing user negatives')
            adj_matrix_slice = self.igraph.adj_matrix[:len(self.user_data), len(self.user_data):]
            user_negatives = []
            for _, row in enumerate(tqdm(range(adj_matrix_slice.shape[0]))):
                negs = np.argwhere(np.array(adj_matrix_slice[row, :].todense()).flatten() == 0).flatten()
                np.random.shuffle(negs)
                negs = negs[:num_negatives] + len(self.user_data)
                negative_edges = np.concatenate((np.repeat(row, negs.shape[0]).reshape(-1, 1), negs.reshape(-1, 1)), axis=1)
                user_negatives.append(negative_edges)
            
            user_negatives = np.array(user_negatives)
            np.save('user_negatives', user_negatives)
            return user_negatives

    def _generate_in_and_oob_negatives(self, positive_edges):
        item_start_id = len(self.user_data)
        pos_edges = np.array(positive_edges)

        negative_edges = []
        for i, (user_id, _) in enumerate(positive_edges):

            # Out of batch negative Sampling
            candidate_item_probs = np.asarray(self.adj_matrix[user_id, item_start_id:].todense()).flatten() == 0
            candidate_item_probs = candidate_item_probs / candidate_item_probs.sum()

            valid_samples = False
            while not valid_samples:
                neg_items = np.random.choice(candidate_item_probs.shape[0], (self.num_neg_samples,), p=candidate_item_probs)     
                for neg_item in neg_items:
                    if (user_id, neg_item + item_start_id) in self.igraph.all_edges:
                        valid_samples = False
                        break

                    valid_samples = True
            
            # In batch negative sampling
            in_batch_candidates = np.concatenate((pos_edges[:i], pos_edges[(i+1):]))[:, 1]
            idxs_to_delete = [idx for idx, candidate in enumerate(in_batch_candidates) if (user_id, candidate) in self.igraph.all_edges]
            valid_candidates = np.delete(in_batch_candidates, idxs_to_delete)
            in_batch_negs = self.rng.choice(valid_candidates, (self.num_neg_samples,), replace=False)

            for neg_item in neg_items:
                negative_edges.append([user_id, neg_item + item_start_id])

            for neg_item in in_batch_negs:
                negative_edges.append([user_id, neg_item])
        
        return negative_edges

    def _get_edges_to_score(self, edges):
        item_start_id = len(self.user_data)
        offsets = []
        edges_to_score = []
        for (user_id, true_edge) in edges:
            current_offset = len(edges_to_score)
            offsets.append(current_offset)

            negatives = np.argwhere(np.array(self.adj_matrix[user_id, item_start_id:item_start_id+20000].todense()).flatten() == 0).flatten() # The true validation and test items will be a part of this array
            edges_to_score += [[user_id, negative + item_start_id] for negative in negatives[:10000]]
            edges_to_score += [[user_id, true_edge]]
        
        return np.array(edges_to_score), np.array(offsets)
    
    def _get_edges_to_score_v2(self, edges):
        offsets = []
        edges_to_score = []
        current_offset = 0
        for (user_id, true_edge_item) in edges:
            offsets.append(current_offset)
            negatives = np.vstack((self.user_negatives[user_id], np.array([user_id, true_edge_item])))
            edges_to_score.append(negatives)
            current_offset += negatives.shape[0]

        return np.vstack(edges_to_score), np.array(offsets)
    
    def _fetch_data(self, edges):
        user_feats = [(user_id, self.user_data[user_id]['location'], self.user_data[user_id]['age']) for user_id, _ in edges]
        item_feats = [(book_id, self.item_data[book_id]['author'], self.item_data[book_id]['date'], self.item_data[book_id]['publisher']) for _, book_id in edges]
        item_title_embeddings = [self.item_data[book_id]['title_embedding'] for _, book_id in edges]

        return user_feats, item_feats, item_title_embeddings

    def __call__(self, positive_edges):
        if self.mode != 'train':
            true_edges = np.array(positive_edges)
            edges_to_score, offsets = self._get_edges_to_score_v2(true_edges)
            return torch.as_tensor(true_edges, dtype=torch.int32), torch.as_tensor(edges_to_score, dtype=torch.int32), torch.as_tensor(offsets, dtype=torch.int32)

        negative_edges = self._generate_in_and_oob_negatives(positive_edges)
        edges = positive_edges + negative_edges
        user_feats, item_feats, item_title_embeddings = self._fetch_data(edges)

        user_feats_by_type = list(zip(*user_feats))
        user_ids = torch.as_tensor(user_feats_by_type[0], dtype=torch.int64)
        user_locations = torch.as_tensor(user_feats_by_type[1], dtype=torch.int64)
        user_ages = torch.as_tensor(user_feats_by_type[2], dtype=torch.int64)

        item_feats_by_type = list(zip(*item_feats))
        book_ids = torch.as_tensor(item_feats_by_type[0], dtype=torch.int64) - len(self.user_data)
        book_authors = torch.as_tensor(item_feats_by_type[1], dtype=torch.int64)
        book_dates = torch.as_tensor(np.asarray(item_feats_by_type[2]), dtype=torch.int64)
        book_publishers = torch.as_tensor(np.asarray(item_feats_by_type[3]), dtype=torch.int64)
        book_title_embeddings = torch.as_tensor(np.asarray(item_title_embeddings), dtype=torch.float32)

        return (user_ids, user_locations, user_ages), (book_ids, book_authors, book_dates, book_publishers, book_title_embeddings)

class BookCrossingInferenceItemsDataset(Dataset):
    def __init__(self, igraph : BookCrossingInteractionGraph) -> None:
        self.all_item_ids = sorted(list(igraph.item_data.keys()))

        self.item_reindexer = {}
        for item_id in self.all_item_ids:
            self.item_reindexer[item_id] = len(self.item_reindexer)

    def __len__(self):
        return len(self.all_item_ids)
    
    def __getitem__(self, index):
        return self.all_item_ids[index]

class BookCrossingItemsCollator:
    def __init__(self, igraph : BookCrossingInteractionGraph) -> None:
        self.igraph = igraph
        self.item_data = igraph.item_data
    
    def __call__(self, batch):
        item_feats = [(movie_id, self.item_data[movie_id]['author'], self.item_data[movie_id]['date'], self.item_data[movie_id]['publisher']) for movie_id in batch]
        item_title_embeddings = [self.item_data[movie_id]['title_embedding'] for movie_id in batch]

        item_feats_by_type = list(zip(*item_feats))
        book_ids = torch.as_tensor(item_feats_by_type[0], dtype=torch.int64)
        zero_indexed_book_ids = book_ids - len(self.igraph.user_data)
        book_authors = torch.as_tensor(item_feats_by_type[1], dtype=torch.int64)
        book_dates = torch.as_tensor(np.asarray(item_feats_by_type[2]), dtype=torch.int64)
        book_publishers = torch.as_tensor(np.asarray(item_feats_by_type[3]), dtype=torch.int64)
        book_title_embeddings = torch.as_tensor(np.asarray(item_title_embeddings), dtype=torch.float32)

        return (book_ids, zero_indexed_book_ids, book_authors, book_dates, book_publishers, book_title_embeddings)

class BookCrossingInferenceUsersDataset(Dataset):
    def __init__(self, igraph : BookCrossingInteractionGraph) -> None:
        self.all_user_ids = sorted(list(igraph.user_data.keys()))
    
    def __len__(self):
        return len(self.all_user_ids)
    
    def __getitem__(self, index):
        return self.all_user_ids[index]

class BookCrossingUsersCollator:
    def __init__(self, igraph : BookCrossingInteractionGraph) -> None:
        self.igraph = igraph
        self.user_data = igraph.user_data

    def __call__(self, batch):
        user_feats = [(user_id, self.user_data[user_id]['location'], self.user_data[user_id]['age']) for user_id in batch]
        user_feats_by_type = list(zip(*user_feats))
        user_feats_by_type = list(zip(*user_feats))
        user_ids = torch.as_tensor(user_feats_by_type[0], dtype=torch.int64)
        user_locations = torch.as_tensor(user_feats_by_type[1], dtype=torch.int64)
        user_ages = torch.as_tensor(user_feats_by_type[2], dtype=torch.int64)

        return (user_ids, user_locations, user_ages)

#endregion