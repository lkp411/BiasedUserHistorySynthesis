from typing import List
import torch
import torch.nn as nn

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




#region MovieLens ------------------------------------
class MovieLensSparseNNUserModel(nn.Module):
    def __init__(self,
        num_user_ids,
        num_user_genders,
        num_user_ages,
        num_user_occupations,
        feat_embed_dim=64,
        output_embed_dim=128,
        combine_op='cat',
    ):
        super(MovieLensSparseNNUserModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_user_ids, embedding_dim=feat_embed_dim)
        self.gender_embeddings = nn.Embedding(num_user_genders, embedding_dim=feat_embed_dim)
        self.age_embeddings = nn.Embedding(num_user_ages, embedding_dim=feat_embed_dim)
        self.occ_embeddings = nn.Embedding(num_user_occupations, embedding_dim=feat_embed_dim)

        self.output_embed_dim = output_embed_dim
        self.combine_op = combine_op
        self.act = nn.GELU()

        self.output_mlp = self._create_output_mlp(4*feat_embed_dim if combine_op == 'cat' else feat_embed_dim, output_embed_dim)
    
    def _create_output_mlp(self, first_layer_dim, output_embed_dim):
        return nn.Sequential(nn.Linear(first_layer_dim, 128), 
                             nn.LayerNorm(128, elementwise_affine=False), 
                             self.act, 
                             nn.Linear(128, 64),
                             nn.LayerNorm(64, elementwise_affine=False),
                             self.act, 
                             nn.Linear(64, output_embed_dim))

    
    def forward(self, user_ids, user_genders, user_ages, user_occs):
        id_embeddings = self.id_embeddings(user_ids)
        gender_embeddings = self.gender_embeddings(user_genders)
        age_embeddings = self.age_embeddings(user_ages)
        occ_embeddings = self.occ_embeddings(user_occs)

        combined_rep = torch.cat([id_embeddings, gender_embeddings, age_embeddings, occ_embeddings], dim=1) if self.combine_op == 'cat' else \
            id_embeddings + gender_embeddings + age_embeddings + occ_embeddings
        
        return self.act(self.output_mlp(combined_rep))

class MovieLensSparseNNItemModel(nn.Module):
    def __init__(self,
        num_movie_ids,
        num_movie_dates,
        num_movie_genres,
        feat_embed_dim=64,
        dense_feat_input_dim=324,
        output_embed_dim=128,
        combine_op='cat'
    ):
        super(MovieLensSparseNNItemModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_movie_ids, feat_embed_dim)
        self.date_embeddings = nn.Embedding(num_movie_dates, feat_embed_dim)
        self.genre_embedding_matrix = nn.Parameter(torch.empty(num_movie_genres, feat_embed_dim))
        nn.init.xavier_normal_(self.genre_embedding_matrix)
        self.dense_transform = nn.Linear(dense_feat_input_dim, feat_embed_dim)

        self.output_embed_dim = output_embed_dim
        self.combine_op = combine_op
        self.act = nn.GELU()

        self.output_mlp = self._create_output_mlp(4*feat_embed_dim if combine_op == 'cat' else feat_embed_dim, output_embed_dim)
        
    
    def _create_output_mlp(self, first_layer_dim, output_embed_dim):
        return nn.Sequential(nn.Linear(first_layer_dim, 128), 
                             nn.LayerNorm(128, elementwise_affine=False), 
                             self.act, 
                             nn.Linear(128, 64),
                             nn.LayerNorm(64, elementwise_affine=False),
                             self.act, 
                             nn.Linear(64, output_embed_dim))

    def forward(self, movie_ids, movie_dates, movie_genres, movie_title_embeddings):
        id_embeddings = self.id_embeddings(movie_ids)
        date_embeddings = self.date_embeddings(movie_dates)
        genre_embeddings = torch.matmul(movie_genres, self.genre_embedding_matrix)
        dense_embeddings = self.act(self.dense_transform(movie_title_embeddings))

        combined_rep = torch.cat([id_embeddings, date_embeddings, genre_embeddings, dense_embeddings], dim=1) if self.combine_op == 'cat' else \
            date_embeddings + genre_embeddings + dense_embeddings
        
        return self.act(self.output_mlp(combined_rep))

class MovieLensSparseNN(nn.Module):
    def __init__(self,
        num_user_ids, num_user_genders, num_user_ages, num_user_occupations,
        num_movie_ids, num_movie_dates, num_movie_genres,
        feat_embed_dim=96,
        dense_feat_embed_dim=384,
        output_embed_dim=192,
        combine_op='cat',
    ):
        super(MovieLensSparseNN, self).__init__()
        self.user_embedding_model = MovieLensSparseNNUserModel(num_user_ids, num_user_genders, num_user_ages, num_user_occupations, 
                                                               feat_embed_dim=feat_embed_dim, 
                                                               output_embed_dim=output_embed_dim,
                                                               combine_op=combine_op)

        self.item_embedding_model = MovieLensSparseNNItemModel(num_movie_ids, num_movie_dates, num_movie_genres,
                                                               feat_embed_dim=feat_embed_dim,
                                                               dense_feat_input_dim=dense_feat_embed_dim,
                                                               output_embed_dim=output_embed_dim,
                                                               combine_op=combine_op)

        self.act = nn.GELU()
        self.scoring_model = DotCompressScoringModel(output_embed_dim, [128, 64])
        


    def forward(self, 
        user_ids = None, user_genders = None, user_ages = None, user_occs = None,
        movie_ids = None, movie_dates = None, movie_genres = None, movie_title_embeddings = None,
        user_embeddings_precomputed = None,
        item_embeddings_precomputed = None,
    ):
        user_embeddings = self.user_embedding_model(user_ids, user_genders, user_ages, user_occs) if user_embeddings_precomputed is None \
            else user_embeddings_precomputed
        
        item_embeddings = self.item_embedding_model(movie_ids, movie_dates, movie_genres, movie_title_embeddings) if item_embeddings_precomputed is None \
            else item_embeddings_precomputed

        return self.act(self.scoring_model(user_embeddings, item_embeddings))
#endregion

#region BookCrossing ----------------------------------
class BookCrossingSparseNNUserModel(nn.Module):
    def __init__(self,
        num_user_ids,
        num_user_locations,
        num_user_ages,
        feat_embed_dim=64,
        output_embed_dim=128,
        combine_op='cat',
    ):
        super(BookCrossingSparseNNUserModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_user_ids, embedding_dim=feat_embed_dim)
        self.location_embeddings = nn.Embedding(num_user_locations, embedding_dim=feat_embed_dim)
        self.age_embeddings = nn.Embedding(num_user_ages, embedding_dim=feat_embed_dim)

        self.output_embed_dim = output_embed_dim
        self.combine_op = combine_op
        self.act = nn.GELU()

        self.output_mlp = self._create_output_mlp(3*feat_embed_dim if combine_op == 'cat' else feat_embed_dim, output_embed_dim)
    
    def _create_output_mlp(self, first_layer_dim, output_embed_dim):
        return nn.Sequential(nn.Linear(first_layer_dim, 128), 
                             nn.LayerNorm(128, elementwise_affine=False), 
                             self.act, 
                             nn.Linear(128, 64),
                             nn.LayerNorm(64, elementwise_affine=False),
                             self.act, 
                             nn.Linear(64, output_embed_dim))

    
    def forward(self, user_ids, user_locations, user_ages):
        id_embeddings = self.id_embeddings(user_ids)
        location_embeddings = self.location_embeddings(user_locations)
        age_embeddings = self.age_embeddings(user_ages)

        combined_rep = torch.cat([id_embeddings, location_embeddings, age_embeddings], dim=1) if self.combine_op == 'cat' else \
            id_embeddings + location_embeddings + age_embeddings
        
        return self.act(self.output_mlp(combined_rep))

class BookCrossingSparseNNItemModel(nn.Module):
    def __init__(self,
        num_book_ids,
        num_book_authors,
        num_book_dates,
        num_book_publishers,
        feat_embed_dim=64,
        dense_feat_input_dim=324,
        output_embed_dim=128,
        combine_op='cat'
    ):
        super(BookCrossingSparseNNItemModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_book_ids, feat_embed_dim)
        self.author_embeddings = nn.Embedding(num_book_authors, feat_embed_dim)
        self.date_embeddings = nn.Embedding(num_book_dates, feat_embed_dim)
        self.publisher_embeddings = nn.Embedding(num_book_publishers, feat_embed_dim)
        self.dense_transform = nn.Linear(dense_feat_input_dim, feat_embed_dim)

        self.output_embed_dim = output_embed_dim
        self.combine_op = combine_op
        self.act = nn.GELU()

        self.output_mlp = self._create_output_mlp(4*feat_embed_dim if combine_op == 'cat' else feat_embed_dim, output_embed_dim)
        
    
    def _create_output_mlp(self, first_layer_dim, output_embed_dim):
        return nn.Sequential(nn.Linear(first_layer_dim, 128), 
                             nn.LayerNorm(128, elementwise_affine=False), 
                             self.act, 
                             nn.Linear(128, 64),
                             nn.LayerNorm(64, elementwise_affine=False),
                             self.act, 
                             nn.Linear(64, output_embed_dim))

    def forward(self, book_ids, book_authors, book_dates, book_publishers, book_title_embeddings):
        id_embeddings = self.id_embeddings(book_ids) # Not used atm
        author_embeddings = self.author_embeddings(book_authors)
        date_embeddings = self.date_embeddings(book_dates)
        publisher_embeddings = self.publisher_embeddings(book_publishers)
        dense_embeddings = self.act(self.dense_transform(book_title_embeddings))

        combined_rep = torch.cat([author_embeddings, date_embeddings, publisher_embeddings, dense_embeddings], dim=1) if self.combine_op == 'cat' else \
            author_embeddings + date_embeddings + publisher_embeddings + dense_embeddings
        
        return self.act(self.output_mlp(combined_rep))

class BookCrossingSparseNN(nn.Module):
    def __init__(self,
        num_user_ids, num_user_locations, num_user_ages,
        num_book_ids, num_book_authors, num_book_dates, num_book_publishers,
        feat_embed_dim=96,
        dense_feat_embed_dim=384,
        output_embed_dim=192,
        combine_op='cat',
    ):
        super(BookCrossingSparseNN, self).__init__()
        self.user_embedding_model = BookCrossingSparseNNUserModel(num_user_ids, num_user_locations, num_user_ages, 
                                                               feat_embed_dim=feat_embed_dim, 
                                                               output_embed_dim=output_embed_dim,
                                                               combine_op=combine_op)

        self.item_embedding_model = BookCrossingSparseNNItemModel(num_book_ids, num_book_authors, num_book_dates, num_book_publishers,
                                                               feat_embed_dim=feat_embed_dim,
                                                               dense_feat_input_dim=dense_feat_embed_dim,
                                                               output_embed_dim=output_embed_dim,
                                                               combine_op=combine_op)
        
        self.act = nn.GELU()
        self.scoring_model = DotCompressScoringModel(output_embed_dim, [128, 64])

    def forward(self, 
        user_ids = None, user_locations = None, user_ages = None,
        book_ids = None, book_authors = None, book_dates = None, book_publishers = None, book_title_embeddings = None,
        user_embeddings_precomputed = None,
        item_embeddings_precomputed = None,
    ):
        user_embeddings = self.user_embedding_model(user_ids, user_locations, user_ages) if user_embeddings_precomputed is None \
            else user_embeddings_precomputed
        
        item_embeddings = self.item_embedding_model(book_ids, book_authors, book_dates, book_publishers, book_title_embeddings) if item_embeddings_precomputed is None \
            else item_embeddings_precomputed
            
        
        return self.act(self.scoring_model(user_embeddings, item_embeddings))
#endregion