from collections import defaultdict
from lib2to3.pytree import convert
from typing import Dict, Hashable, List
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import math
from tqdm import tqdm
import pycountry


def convert_to_categorical(input_list: List[Hashable], offset=0) -> Dict[Hashable, int]:
    unique_items = list(set(input_list))
    out = {item : i + offset for i, item in enumerate(unique_items)}
    return out

def read_movielens(datasets_dir=None):
    GENRE_ID = {
        "Action" : 0,
        "Adventure" : 1,
        "Animation" : 2,
        "Children's" : 3,
        "Comedy" : 4,
        "Crime" : 5, 
        "Documentary" : 6,
        "Drama" : 7,
        "Fantasy" : 8,
        "Film-Noir" : 9,
        "Horror" : 10, 
        "Musical" : 11,
        "Mystery" : 12,
        "Romance" : 13,
        "Sci-Fi" : 14,
        "Thriller" : 15,
        "War" : 16,
        "Western" : 17
    }

    AGE_ID = {
        1 : 0,
        18 : 1, 
        25 : 2,
        35 : 3,
        45 : 4,
        50 : 5,
        56 : 6
    }

    data_dir = datasets_dir + '/ml-1m'
    movie_data_file = data_dir + '/movies.dat'
    user_data_file = data_dir + '/users.dat'
    ratings_data_file = data_dir + '/ratings.dat'

    # Read User Data
    user_data = {}
    with open(user_data_file, 'r', encoding='latin-1') as f:
        for line in f:
            user_info = line.split("::")
            user_id = int(user_info[0])
            gender_feat = 0 if user_info[1] == 'M' else 1
            age_feat = AGE_ID[int(user_info[2])]
            occupation_feat = int(user_info[3])

            user_data[user_id] = {}
            user_data[user_id]['gender'] = gender_feat
            user_data[user_id]['age'] = age_feat
            user_data[user_id]['occupation'] = occupation_feat
    
    user_id_reindexer = {old_idx : new_idx for new_idx, old_idx in enumerate(user_data)}
    reindexed_user_data = {user_id_reindexer[old_idx] : user_data[old_idx] for old_idx in user_data}
    num_users = len(reindexed_user_data)

    # Read Movie Data
    movie_data = {}
    movie_titles = []
    dates = []  
    with open(movie_data_file, 'r', encoding='latin-1') as f:
        for line in f:
            movie_info = line.split("::")

            movie_id = int(movie_info[0])
            title = movie_info[1][:-6].strip()
            date = int(movie_info[1][-6:][1:-1])
            genres = list(map(lambda genre : GENRE_ID[genre], movie_info[-1].strip().split('|')))
            one_hot = np.zeros((len(GENRE_ID),), dtype=np.float)
            one_hot[genres] = 1

            movie_data[movie_id] = {}
            movie_data[movie_id]['genres'] = one_hot
            movie_titles.append(title)
            dates.append(date)


    movie_id_reindexer = {old_idx : new_idx for new_idx, old_idx in enumerate(movie_data)}
    reindexed_movie_data = {movie_id_reindexer[old_idx] : movie_data[old_idx] for old_idx in movie_data}

    unique_dates = np.unique(dates).tolist()
    date_featurizer = {date : i for i, date in enumerate(unique_dates)}
    date_feats = list(map(lambda date : date_featurizer[date], dates))

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    title_embeddings = model.encode(movie_titles)

    for idx in reindexed_movie_data:
        reindexed_movie_data[idx]['date'] = date_feats[idx]
        reindexed_movie_data[idx]['title_embedding'] = title_embeddings[idx]

    movie_data_with_scaled_idx = {idx + num_users : reindexed_movie_data[idx] for idx in reindexed_movie_data} # Item IDs start at num_users and go to (num_users + num_items - 1)
    del reindexed_movie_data, movie_data

    # Read Interactions
    ratings = defaultdict(list)
    with open(ratings_data_file, 'r', encoding='latin-1') as f:
        for line in f:
            interaction_data = line.split("::")

            user_id = user_id_reindexer[int(interaction_data[0])]
            item_id = movie_id_reindexer[int(interaction_data[1])] + num_users
            rating = int(interaction_data[2])
            time_stamp = int(interaction_data[-1].strip())

            ratings[user_id].append((item_id, rating, time_stamp, int(interaction_data[0]), int(interaction_data[1])))
    return reindexed_user_data, movie_data_with_scaled_idx, ratings, len(unique_dates)


def read_bx(datasets_dir=None):
    data_dir = datasets_dir + '/BookCrossing'
    book_data_file = data_dir + '/Books.csv'
    user_data_file = data_dir + '/Users.csv'
    ratings_data_file = data_dir + '/Ratings.csv'

    df_books = pd.read_csv(book_data_file, delimiter=";", encoding='utf-8')
    df_users = pd.read_csv(user_data_file, delimiter=";", encoding='utf-8')
    df_ratings = pd.read_csv(ratings_data_file, delimiter=";", encoding='utf-8')

    all_countries = set(list(map(lambda x : x.name.lower(), pycountry.countries)))

    book_isbn_ids = df_books['ISBN'].to_numpy().astype(str).tolist()
    book_titles = df_books['Title'].to_numpy().astype(str).tolist()
    book_authors = df_books['Author'].to_numpy().astype(str).tolist()
    book_years = df_books['Year'].to_numpy().astype(int).tolist()
    book_publishers = df_books['Publisher'].to_numpy().astype(str).tolist()
    
    isbn_set = set(book_isbn_ids)

    # Read ratings
    ratings = defaultdict(list)
    user_ids_in_edges = df_ratings['User-ID'].to_numpy().astype(int).tolist()
    isbn_ids_in_edges = df_ratings['ISBN'].to_numpy().astype(str)
    edge_ratings = df_ratings['Rating'].to_numpy().astype(int)
    all_items = set()

    ############ First level filtering of edges################
    for i in range(len(user_ids_in_edges)):
        user_id = user_ids_in_edges[i]
        isbn_id = isbn_ids_in_edges[i]
        rating = edge_ratings[i]

        if isbn_id not in isbn_set:
            continue

        ratings[user_id].append((isbn_id, rating)) # (item_id, rating, original user id, isbn id)
        all_items.add(isbn_id)
    ############################################################
    
    ################ Filtering of users ########################
    all_users = set(ratings.keys())
    user_ids = []
    user_locations = []
    user_ages = []
    for _, user in enumerate(tqdm(set(all_users))):
        df_row = df_users.loc[df_users['User-ID'] == user]
        location = df_row['Location'].item()
        age = df_row['Age'].item()

        if math.isnan(age) or (age < 5 or age > 110):
            all_users.remove(user)
            continue

        try:
            country = location.split(',')[2].strip().lower()
            if ((country == '') or (country not in all_countries)) and country != 'usa':
                all_users.remove(user)
                continue
        except:
            all_users.remove(user)
            continue

        user_ids.append(user)
        user_locations.append(country)
        user_ages.append(int(age))

    user_id_reindexer =  convert_to_categorical(user_ids) 
    user_location_to_idx = convert_to_categorical(user_locations)
    user_ages_to_idx = convert_to_categorical(user_ages)
    #################################################################

    #################### Second level filtering of edges #############
    ratings_filt = defaultdict(list)
    all_items = set()
    for user_id in all_users:
        for (isbn_id, rating) in ratings[user_id]:
            ratings_filt[user_id].append((isbn_id, rating))
            all_items.add(isbn_id)
    
    ratings = ratings_filt
    ###################################################################

    user_data = {}
    for i in range(len(user_ids)):
        reindexed_user_id = user_id_reindexer[user_ids[i]]
        location_id = user_location_to_idx[user_locations[i]]
        age_id = user_ages_to_idx[user_ages[i]]

        user_data[reindexed_user_id] = {'location' : location_id, 'age' : age_id}

    # Read Book info
    book_isbn_ids_filt, book_titles_filt, book_authors_filt, book_years_filt, book_publishers_filt = [], [], [], [], []
    isbn_set = set()
    for i in range(len(book_isbn_ids)):
        isbn_id = book_isbn_ids[i]
        if (isbn_id in isbn_set) or (isbn_id not in all_items):
            continue

        book_isbn_ids_filt.append(isbn_id)
        book_titles_filt.append(book_titles[i])
        book_authors_filt.append(book_authors[i].lower())
        book_years_filt.append(book_years[i])
        book_publishers_filt.append(book_publishers[i].lower())
        isbn_set.add(isbn_id)

    book_isbn_ids = book_isbn_ids_filt
    book_titles = book_titles_filt
    book_authors = book_authors_filt
    book_years = book_years_filt
    book_publishers = book_publishers_filt

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    title_embeddings = model.encode(book_titles, batch_size=1536, show_progress_bar=True)

    isbn_to_idx_map = convert_to_categorical(book_isbn_ids, offset=len(user_data))
    authors_to_idx_map = convert_to_categorical(book_authors) 
    years_to_idxs_map = convert_to_categorical(book_years) 
    publisher_to_idxs_map = convert_to_categorical(book_publishers) 

    item_data = {}
    for i in range(len(book_isbn_ids)):
        book_id = isbn_to_idx_map[book_isbn_ids[i]]
        title_embedding = title_embeddings[i]
        author = authors_to_idx_map[book_authors[i]]
        year = years_to_idxs_map[book_years[i]]
        publisher = publisher_to_idxs_map[book_publishers[i]]

        item_data[book_id] = {'title_embedding' : title_embedding, 'author' : author, 'date' : year, 'publisher' : publisher}

    
    reindexed_ratings = defaultdict(list)
    for user_id in all_users:
        for (isbn_id, rating) in ratings[user_id]:
            reindexed_ratings[user_id_reindexer[user_id]].append((isbn_to_idx_map[isbn_id], rating, user_id, isbn_id))
    
    return user_data, item_data, reindexed_ratings, len(user_location_to_idx), len(user_ages_to_idx), len(authors_to_idx_map), len(years_to_idxs_map), len(publisher_to_idxs_map)


if __name__ == '__main__':
    read_bx(datasets_dir='/home/keshav/datasets')