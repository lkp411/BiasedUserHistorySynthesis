import torch
import torch.nn as nn
from model.setmembershipmodels import (
    NeighborAveragingSetRepresentationModel,
    MultiHeadAttentionSetRepresentationModel,
    RecurrentSetRepresentationModel
)

from model.misc import (
    DotProduct,
    ItemEmbeddingModel, 
    DotCompressScoringModel
)

#region MovieLens ---------------------------------------------------------------------------------------
class MovieLensSparseNNUserModel(nn.Module):
    def __init__(self,
        num_user_ids,
        num_user_genders,
        num_user_ages,
        num_user_occupations,
        feat_embed_dim=64,
    ):
        super(MovieLensSparseNNUserModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_user_ids, embedding_dim=feat_embed_dim)
        self.gender_embeddings = nn.Embedding(num_user_genders, embedding_dim=feat_embed_dim)
        self.age_embeddings = nn.Embedding(num_user_ages, embedding_dim=feat_embed_dim)
        self.occ_embeddings = nn.Embedding(num_user_occupations, embedding_dim=feat_embed_dim)
    
    def forward(self, user_ids, user_genders, user_ages, user_occs):
        id_embeddings = self.id_embeddings(user_ids)
        gender_embeddings = self.gender_embeddings(user_genders)
        age_embeddings = self.age_embeddings(user_ages)
        occ_embeddings = self.occ_embeddings(user_occs)

        combined_rep = torch.cat([id_embeddings, gender_embeddings, age_embeddings, occ_embeddings], dim=1)
        return combined_rep

class MovieLensSparseNNItemModel(nn.Module):
    def __init__(self,
        num_movie_ids,
        num_movie_dates,
        num_movie_genres,
        feat_embed_dim=64,
        dense_feat_input_dim=384,
    ):
        super(MovieLensSparseNNItemModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_movie_ids, feat_embed_dim)
        self.date_embeddings = nn.Embedding(num_movie_dates, feat_embed_dim)
        self.genre_embedding_matrix = nn.Parameter(torch.empty(num_movie_genres, feat_embed_dim))
        nn.init.xavier_normal_(self.genre_embedding_matrix)
        self.dense_transform = nn.Linear(dense_feat_input_dim, feat_embed_dim)

        self.register_parameter("genre_embedding_matrix", self.genre_embedding_matrix)

    def forward(self, movie_ids, movie_dates, movie_genres, movie_title_embeddings):
        id_embeddings = self.id_embeddings(movie_ids)
        date_embeddings = self.date_embeddings(movie_dates)        
        genre_embeddings = torch.matmul(movie_genres, self.genre_embedding_matrix)
        dense_embeddings = self.dense_transform(movie_title_embeddings)

        combined_rep = torch.cat([id_embeddings, date_embeddings, genre_embeddings, dense_embeddings], dim=1)
        return combined_rep

class MovieLensSparseNN(nn.Module):
    def __init__(self,
        num_user_ids, num_user_genders, num_user_ages, num_user_occupations,
        num_movie_ids, num_movie_dates, num_movie_genres,
        feat_embed_dim=96,
        dense_feat_embed_dim=384,
        set_hidden_dims=[512, 256, 128],
        item_embed_hidden_dims=[256, 128],
        scoring_function_hidden_dims=[128, 64],
        projection_dim=64,
        set_embed_dim=128,
        seq_len=15,
        recurr_hidden_size=32,
        recurr_num_layers=2,
        nhead=1,
        set_embedding_model='mean',
        recurr_mode='gru',
        scoring_model='dot_compress'
    ):
        super(MovieLensSparseNN, self).__init__()
        self.feat_embed_dim = feat_embed_dim
        self.set_hidden_dims = set_hidden_dims
        self.item_embed_hidden_dims = item_embed_hidden_dims
        self.scoring_function_hidden_dims = scoring_function_hidden_dims
        self.set_embed_dim = set_embed_dim

        self.user_feature_model = MovieLensSparseNNUserModel(num_user_ids, num_user_genders, num_user_ages, num_user_occupations, feat_embed_dim=feat_embed_dim)
        self.item_feature_model = MovieLensSparseNNItemModel(num_movie_ids, num_movie_dates, num_movie_genres, feat_embed_dim=feat_embed_dim, dense_feat_input_dim=dense_feat_embed_dim)
        
        if set_embedding_model == 'mean':
            self.set_embedding_model = NeighborAveragingSetRepresentationModel(4*feat_embed_dim, 4*feat_embed_dim, set_embedding_dim=set_embed_dim, hidden_dims=set_hidden_dims)
        elif set_embedding_model == 'mha':
            self.set_embedding_model = MultiHeadAttentionSetRepresentationModel(4*feat_embed_dim, 
                                                                                4*feat_embed_dim, 
                                                                                projection_dim=projection_dim, 
                                                                                set_embedding_dim=set_embed_dim, 
                                                                                hidden_dims=set_hidden_dims,
                                                                                num_heads=nhead)
        elif set_embedding_model == 'recurr':
            self.set_embedding_model = RecurrentSetRepresentationModel(seq_len, 
                                                                       4*feat_embed_dim, 
                                                                       4*feat_embed_dim, 
                                                                       recurr_hidden_size, 
                                                                       recurr_num_layers, 
                                                                       set_embedding_dim=set_embed_dim, 
                                                                       hidden_dims=set_hidden_dims,
                                                                       mode=recurr_mode)
        else:
            raise NotImplementedError

        self.item_embedding_model = ItemEmbeddingModel(4*feat_embed_dim, item_embed_hidden_dims, set_embed_dim)

        self.scoring_model = DotCompressScoringModel(set_embed_dim, scoring_function_hidden_dims) if scoring_model == 'dot_compress' else DotProduct()
        
    def forward(self, 
        user_ids = None, user_genders = None, user_ages = None, user_occs = None,
        movie_ids = None, movie_dates = None, movie_genres = None, movie_title_embeddings = None,
        neighbor_movie_ids = None, neighbor_movie_dates = None, neighbor_movie_genres = None, neighbor_movie_title_embeddings = None,
        set_embeddings_precomputed = None,
        item_embeddings_precomputed = None,
    ):
        if set_embeddings_precomputed is None:
            user_feats = self.user_feature_model(user_ids, user_genders, user_ages, user_occs)
            neighbor_item_feats = self.item_feature_model(neighbor_movie_ids, neighbor_movie_dates, neighbor_movie_genres, neighbor_movie_title_embeddings)
            set_embeds = self.set_embedding_model(user_feats, neighbor_item_feats.view(user_feats.shape[0], -1, neighbor_item_feats.shape[1]))
        else:
            set_embeds = set_embeddings_precomputed

        if item_embeddings_precomputed is None:
            item_feats = self.item_feature_model(movie_ids, movie_dates, movie_genres, movie_title_embeddings)
            item_embeds = self.item_embedding_model(item_feats)
        else:
            item_embeds = item_embeddings_precomputed

        score = self.scoring_model(set_embeds, item_embeds)
        return score
#endregion

#region BookCrossing -------------------------------------------------------------------------
class BookCrossingSparseNNUserModel(nn.Module):
    def __init__(self,
        num_user_ids,
        num_user_locations,
        num_user_ages,
        feat_embed_dim=64
    ):
        super(BookCrossingSparseNNUserModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_user_ids, embedding_dim=feat_embed_dim)
        self.location_embeddings = nn.Embedding(num_user_locations, embedding_dim=feat_embed_dim)
        self.age_embeddings = nn.Embedding(num_user_ages, embedding_dim=feat_embed_dim)
    
    def forward(self, user_ids, user_locations, user_ages):
        id_embeddings = self.id_embeddings(user_ids)
        location_embeddings = self.location_embeddings(user_locations)
        age_embeddings = self.age_embeddings(user_ages)

        combined_rep = torch.cat([id_embeddings, location_embeddings, age_embeddings], dim=1)
        return combined_rep

class BookCrossingSparseNNItemModel(nn.Module):
    def __init__(self,
        num_book_ids,
        num_book_authors,
        num_book_dates,
        num_book_publishers,
        feat_embed_dim=64,
        dense_feat_input_dim=384,
    ):
        super(BookCrossingSparseNNItemModel, self).__init__()
        self.id_embeddings = nn.Embedding(num_book_ids, feat_embed_dim)
        self.author_embeddings = nn.Embedding(num_book_authors, feat_embed_dim)
        self.date_embeddings = nn.Embedding(num_book_dates, feat_embed_dim)
        self.publisher_embeddings = nn.Embedding(num_book_publishers, feat_embed_dim)
        self.dense_transform = nn.Linear(dense_feat_input_dim, feat_embed_dim)

    def forward(self, book_ids, book_authors, book_dates, book_publishers, book_title_embeddings):
        id_embeddings = self.id_embeddings(book_ids) # Not used atm
        author_embeddings = self.author_embeddings(book_authors)
        date_embeddings = self.date_embeddings(book_dates)
        publisher_embeddings = self.publisher_embeddings(book_publishers)
        dense_embeddings = self.dense_transform(book_title_embeddings)

        combined_rep = torch.cat([author_embeddings, date_embeddings, publisher_embeddings, dense_embeddings], dim=1)
        return combined_rep

class BookCrossingSparseNN(nn.Module):
    def __init__(self,
        num_user_ids, num_user_locations, num_user_ages,
        num_book_ids, num_book_authors, num_book_dates, num_book_publishers,
        feat_embed_dim=96,
        dense_feat_embed_dim=384,
        set_hidden_dims=[512, 256, 128],
        item_embed_hidden_dims=[256, 128],
        scoring_function_hidden_dims=[128, 64],
        projection_dim=64,
        set_embed_dim=128,
        seq_len=15,
        recurr_hidden_size=32,
        recurr_num_layers=2,
        recurr_mode='gru',
        nhead=1,
        set_embedding_model='mean'
    ):
        super(BookCrossingSparseNN, self).__init__()
        self.feat_embed_dim = feat_embed_dim
        self.set_hidden_dims = set_hidden_dims
        self.item_embed_hidden_dims = item_embed_hidden_dims
        self.scoring_function_hidden_dims = scoring_function_hidden_dims
        self.set_embed_dim = set_embed_dim

        self.user_feature_model = BookCrossingSparseNNUserModel(num_user_ids, num_user_locations, num_user_ages, feat_embed_dim=feat_embed_dim)
        self.item_feature_model = BookCrossingSparseNNItemModel(num_book_ids, num_book_authors, num_book_dates, num_book_publishers, feat_embed_dim=feat_embed_dim, dense_feat_input_dim=dense_feat_embed_dim)
        
        if set_embedding_model == 'mean':
            self.set_embedding_model = NeighborAveragingSetRepresentationModel(3*feat_embed_dim, 4*feat_embed_dim, set_embedding_dim=set_embed_dim, hidden_dims=set_hidden_dims)
        elif set_embedding_model == 'mha':
            self.set_embedding_model = MultiHeadAttentionSetRepresentationModel(3*feat_embed_dim, 
                                                                                4*feat_embed_dim, 
                                                                                projection_dim=projection_dim, 
                                                                                set_embedding_dim=set_embed_dim, 
                                                                                hidden_dims=set_hidden_dims,
                                                                                num_heads=nhead)
        elif set_embedding_model == 'recurr':
            self.set_embedding_model = RecurrentSetRepresentationModel(seq_len, 
                                                                       3*feat_embed_dim, 
                                                                       4*feat_embed_dim, 
                                                                       recurr_hidden_size, 
                                                                       recurr_num_layers, 
                                                                       set_embedding_dim=set_embed_dim, 
                                                                       hidden_dims=set_hidden_dims,
                                                                       mode=recurr_mode)
        else:
            raise NotImplementedError

        self.item_embedding_model = ItemEmbeddingModel(4*feat_embed_dim, item_embed_hidden_dims, set_embed_dim)

        self.scoring_model = DotCompressScoringModel(set_embed_dim, scoring_function_hidden_dims)

    def forward(self, 
        user_ids = None, user_locations = None, user_ages = None,
        book_ids = None, book_authors = None, book_dates = None, book_publishers = None, book_title_embeddings = None,
        neighbor_book_ids = None, neighbor_book_authors = None, neighbor_book_dates = None, neighbor_book_publishers = None, neighbor_book_title_embeddings = None,
        set_embeddings_precomputed = None,
        item_embeddings_precomputed = None,
    ):
        if set_embeddings_precomputed is None:
            user_feats = self.user_feature_model(user_ids, user_locations, user_ages)
            neighbor_item_feats = self.item_feature_model(neighbor_book_ids, neighbor_book_authors, neighbor_book_dates, neighbor_book_publishers, neighbor_book_title_embeddings)
            set_embeds = self.set_embedding_model(user_feats, neighbor_item_feats.view(user_feats.shape[0], -1, neighbor_item_feats.shape[1]))
        else:
            set_embeds = set_embeddings_precomputed

        if item_embeddings_precomputed is None:
            item_feats = self.item_feature_model(book_ids, book_authors, book_dates, book_publishers, book_title_embeddings)
            item_embeds = self.item_embedding_model(item_feats)
        else:
            item_embeds = item_embeddings_precomputed

        score = self.scoring_model(set_embeds, item_embeds)
        return score
#endregion