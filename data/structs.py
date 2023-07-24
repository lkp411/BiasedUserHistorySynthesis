from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
import scipy.sparse
from random import random, shuffle
from data.datareader import read_bx
import random

class InteractionGraph:
    def __init__(self, user_data, item_data, interactions) -> None:
        self.user_data = user_data
        self.item_data = item_data
        self.interactions = interactions
        self.train_edges, self.validation_edges, self.test_edges = [], [], []
        self.adj_matrix: scipy.sparse.dok_matrix = None

    def split_statistics(self):
        training_items = set(self.train_edges[:, 1])
        validation_items = set(self.validation_edges[:, 1])
        test_items = set(self.test_edges[:, 1])

        print("Total number of items = {}".format(len(self.item_data)))
        print("Number of items present across training edges = {}".format(len(training_items)))
        print("Number of items present across val edges = {}".format(len(validation_items)))
        print("Number of items present across test edges = {}".format(len(test_items)))
        print("Average item degree = {}".format(np.mean(self.item_degrees)))
        print("Average user degree = {}".format(np.mean(self.user_degrees)))

        train_val_common_items = training_items.intersection(validation_items)
        train_test_common_items = training_items.intersection(test_items)

        print('Number of items common between train and validation edges = {}'.format(len(train_val_common_items)))
        print('Number of items common between train and test edges = {}'.format(len(train_test_common_items)))

        validation_items = np.array(list(validation_items))
        test_items = np.array(list(test_items))

        num_cold_items_in_val = np.sum(self.is_cold[validation_items])
        num_cold_items_in_test = np.sum(self.is_cold[test_items])

        print('Number of cold items in validation set = {}'.format(num_cold_items_in_val))
        print('Number of cold items in test set = {}'.format(num_cold_items_in_test))


    def create_bipartite_graph(self):
        num_nodes = len(self.user_data) + len(self.item_data) # Num users + num items 
        self.adj_matrix = scipy.sparse.dok_matrix((num_nodes, num_nodes), dtype=bool)  # TODO: Maybe we can optimize with lower precision data types
        
        for edge in self.train_edges:
            self.adj_matrix[edge[0], edge[1]] = 1
            self.adj_matrix[edge[1], edge[0]] = 1

        self.adj_matrix = self.adj_matrix.tocsr()
    
    def compute_tail_distribution(self, warm_threshold):
        self.is_cold = np.zeros((self.adj_matrix.shape[0]), dtype=np.bool)
        self.start_item_id = len(self.user_data)

        self.user_degrees = np.array(self.adj_matrix[:self.start_item_id].sum(axis=1)).flatten()
        self.item_degrees = np.array(self.adj_matrix[self.start_item_id:].sum(axis=1)).flatten()

        cold_items = np.argsort(self.item_degrees)[:int((1 - warm_threshold) * len(self.item_degrees))] + self.start_item_id
        self.is_cold[cold_items] = True

    def create_data_split(self):
        raise NotImplementedError()
    

class MovielensInteractionGraph(InteractionGraph):
    def __init__(self, user_data, item_data, interactions, warm_threshold=0.2) -> None:
        super().__init__(user_data, item_data, interactions)
        self.create_data_split()
        self.create_bipartite_graph()
        assert (warm_threshold < 1.0 and warm_threshold > 0.0)
        self.warm_threshold = warm_threshold
        self.compute_tail_distribution()
    
    def create_data_split(self):
        # Leave one out validation - for each user the latest interaction is a test item and the second latest item is the validation item
        print('Creating data split')
        self.all_edges = set()
        self.train_edges_set = set()
        for user_id in tqdm(self.interactions):
            sorted_interactions = sorted(self.interactions[user_id], key=lambda x : x[2])
            test_edge = [user_id, sorted_interactions[-1][0]]
            val_edge = [user_id, sorted_interactions[-2][0]]
            self.all_edges.add((user_id, sorted_interactions[-2][0]))

            train_edges = [[user_id, interaction[0]] for interaction in sorted_interactions[:-2]]
            for interaction in sorted_interactions[:-2]:
                self.all_edges.add((user_id, interaction[0]))
                self.train_edges_set.add((user_id, interaction[0]))           

            self.train_edges += train_edges
            self.validation_edges.append(val_edge)
            self.test_edges.append(test_edge)
        
        self.train_edges = np.array(self.train_edges)
        self.validation_edges = np.array(self.validation_edges)
        self.test_edges = np.array(self.test_edges)
    
    def compute_tail_distribution(self):
        return super().compute_tail_distribution(self.warm_threshold)

    def __getitem__(self, user_id):
        assert user_id < len(self.user_data), "User ID out of bounds"
        assert isinstance(self.adj_matrix, scipy.sparse.csr_matrix), "Bipartite graph not created: must call create_bipartite_graph first"
        return np.array(self.adj_matrix[user_id, self.start_item_id:].todense()).flatten().nonzero()[0] + self.start_item_id

    def create_omega_k(self):
        item_as_key = defaultdict(list)
        omega_k = defaultdict(list)
        for user_id in tqdm(self.interactions):
            items = [(interaction[0], interaction[4]) for interaction in self.interactions[user_id]] # (reindexed_item_id, original_item_id)
            original_user_id = self.interactions[user_id][0][3]
            for item in items:
                if (user_id, item[0]) in self.train_edges_set:
                    item_as_key[(item[0], item[1])].append((user_id, original_user_id)) # User ID, Ratings (implicit)

        start_item_id = len(self.user_data)
        item_cold_status = self.is_cold[start_item_id:]
        cold_item_idxs = item_cold_status.nonzero()[0]
        max_degree = self.item_degrees[cold_item_idxs].max()

        for pair in item_as_key:
            if len(item_as_key[pair]) > max_degree:
                random.shuffle(item_as_key[pair])
                item_as_key[pair] = item_as_key[pair][:max_degree]
        
        for (reindexed_item_id, original_item_id) in item_as_key:
            user_list = item_as_key[(reindexed_item_id, original_item_id)]
            for (reindexed_user_id, original_user_id) in user_list:
                if (reindexed_user_id, reindexed_item_id) in self.train_edges_set:
                    omega_k[original_user_id].append(original_item_id)

        
        with open("omega_k.json", "w") as f:
            json.dump(omega_k, f)

    def create_json_files(self):
        warm_state = defaultdict(list)
        warm_state_y = defaultdict(list)
        item_cold_state_val = defaultdict(list)
        item_cold_state_val_y = defaultdict(list)
        item_warm_state_val = defaultdict(list)
        item_warm_state_val_y = defaultdict(list)
        item_cold_state_test = defaultdict(list)
        item_cold_state_test_y = defaultdict(list)
        item_warm_state_test = defaultdict(list)
        item_warm_state_test_y = defaultdict(list)

        for user_id in tqdm(self.interactions):
            sorted_interactions = sorted(self.interactions[user_id], key=lambda x : x[2])
            # Add val item
            original_user_id = sorted_interactions[-2][3]
            original_item_id = sorted_interactions[-2][4]
            reindexed_item_id = sorted_interactions[-2][0]
            rating = sorted_interactions[-2][1]

            if self.is_cold[reindexed_item_id]:
                item_cold_state_val[str(original_user_id)].append(str(original_item_id))
                item_cold_state_val_y[str(original_user_id)].append(rating)
            else:
                item_warm_state_val[str(original_user_id)].append(str(original_item_id))
                item_warm_state_val_y[str(original_user_id)].append(rating)

            # Add test item
            original_user_id = sorted_interactions[-1][3]
            original_item_id = sorted_interactions[-1][4]
            reindexed_item_id = sorted_interactions[-1][0]
            rating = sorted_interactions[-1][1]

            if self.is_cold[reindexed_item_id]:
                item_cold_state_test[str(original_user_id)].append(str(original_item_id))
                item_cold_state_test_y[str(original_user_id)].append(rating)
            else:
                item_warm_state_test[str(original_user_id)].append(str(original_item_id))
                item_warm_state_test_y[str(original_user_id)].append(rating)               

            for interaction in sorted_interactions[:-2]:
                original_user_id = interaction[3]
                original_item_id = interaction[4]
                rating = interaction[1]
                warm_state[str(original_user_id)].append(str(original_item_id))
                warm_state_y[str(original_user_id)].append(rating)
        
        with open("warm_state.json", "w") as f:
            json.dump(warm_state, f)

        with open("warm_state_y.json", "w") as f:
            json.dump(warm_state_y, f)

        with open("item_warm_state_val.json", "w") as f:
            json.dump(item_warm_state_val, f)
        
        with open("item_warm_state_val_y.json", "w") as f:
            json.dump(item_warm_state_val_y, f)
        
        with open("item_cold_state_val.json", "w") as f:
            json.dump(item_cold_state_val, f)
        
        with open("item_cold_state_val_y.json", "w") as f:
            json.dump(item_cold_state_val_y, f)

        with open("item_warm_state_test.json", "w") as f:
            json.dump(item_warm_state_test, f)
        
        with open("item_warm_state_test_y.json", "w") as f:
            json.dump(item_warm_state_test_y, f)
        
        with open("item_cold_state_test.json", "w") as f:
            json.dump(item_cold_state_test, f)
        
        with open("item_cold_state_test_y.json", "w") as f:
            json.dump(item_cold_state_test_y, f)



class BookCrossingInteractionGraph(InteractionGraph):
    def __init__(self, user_data, item_data, interactions, warm_threshold=0.001) -> None:
        super().__init__(user_data, item_data, interactions)
        self.create_data_split()
        self.create_bipartite_graph()
        assert (warm_threshold < 1.0 and warm_threshold > 0.0)
        self.warm_threshold = warm_threshold
        self.compute_tail_distribution()


    def create_data_split(self):
        print('Creating data split')
        self.all_edges = set()
        for user_id in tqdm(self.interactions):
            if len(self.interactions[user_id]) < 3:
                train_edges = [[user_id, interaction[0]] for interaction in self.interactions[user_id]]
                for interaction in self.interactions[user_id]:
                    self.all_edges.add((user_id, interaction[0]))
            else:
                shuffle(self.interactions[user_id])
                test_edge = [user_id, self.interactions[user_id][0][0]]
                val_edge = [user_id, self.interactions[user_id][1][0]]
                self.all_edges.add((user_id, self.interactions[user_id][1][0]))

                train_edges = [[user_id, interaction[0]] for interaction in self.interactions[user_id][2:]]
                for interaction in self.interactions[user_id][2:]:
                    self.all_edges.add((user_id, interaction[0]))
                
                self.validation_edges.append(val_edge)
                self.test_edges.append(test_edge)

            self.train_edges += train_edges
            
        
        self.train_edges = np.array(self.train_edges)
        self.validation_edges = np.array(self.validation_edges)
        self.test_edges = np.array(self.test_edges)

    def compute_tail_distribution(self):
        return super().compute_tail_distribution(self.warm_threshold)