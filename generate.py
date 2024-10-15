import numpy as np
import argparse
import pandas as pd
import tensorflow as tf
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate, BatchNormalization, Dropout, MultiHeadAttention, Dot, Add, Lambda  # Added Add, Lambda
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Softmax, Multiply
from sklearn.metrics import mean_squared_error
from math import sqrt
from tensorflow.keras.activations import swish
from tensorflow.keras.regularizers import l2



namesngenre = np.load('namesngenre.npy', allow_pickle=True)

def prepare_genres_encoding(namesngenre):
    all_genres = set()
    for movie, genre_str in namesngenre:
        genres = genre_str.split('|') 
        all_genres.update(genres)

    genre_list = sorted(list(all_genres))

    genre_to_index = {genre: idx for idx, genre in enumerate(genre_list)}


    movie_genre_encoding = np.zeros((len(namesngenre), len(genre_list)))
    for idx, (movie, genre_str) in enumerate(namesngenre):
        genres = genre_str.split('|')
        for genre in genres:
            genre_idx = genre_to_index[genre]
            movie_genre_encoding[idx, genre_idx] = 1

    return movie_genre_encoding, genre_list

movie_genre_encoding, genre_list = prepare_genres_encoding(namesngenre)

# Build NCF model with MF, biases, and genre embeddings
def build_ncf_model_genres_with_mf_and_bias(num_users, num_items, num_genres, embedding_dim, global_average, reg_lambda=1e-5):
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    genre_input = Input(shape=(num_genres,), name='genre_input')

    user_embedding = Embedding(input_dim=num_users, output_dim=embedding_dim, input_length=1, name='user_embedding', embeddings_regularizer=l2(reg_lambda))(user_input)
    user_embedding = Flatten()(user_embedding)

    item_embedding = Embedding(input_dim=num_items, output_dim=embedding_dim, input_length=1, name='item_embedding', embeddings_regularizer=l2(reg_lambda))(item_input)
    item_embedding = Flatten()(item_embedding)

    genre_embedding = Dense(embedding_dim, activation='relu', name='genre_embedding', kernel_regularizer=l2(reg_lambda))(genre_input)

    user_bias = Embedding(input_dim=num_users, output_dim=1, input_length=1, name='user_bias', embeddings_initializer='zeros', embeddings_regularizer=l2(reg_lambda))(user_input)
    user_bias = Flatten()(user_bias)

    item_bias = Embedding(input_dim=num_items, output_dim=1, input_length=1, name='item_bias', embeddings_initializer='zeros', embeddings_regularizer=l2(reg_lambda))(item_input)
    item_bias = Flatten()(item_bias)

    mf_interaction = Dot(axes=1, name='mf_interaction')([user_embedding, item_embedding])

    prediction = Add(name='prediction')([mf_interaction, user_bias, item_bias, Lambda(lambda x: tf.fill([tf.shape(x)[0], 1], global_average), name='global_average')(mf_interaction)])

    concatenated = Concatenate(name='concatenated_features')([user_embedding, genre_embedding, item_embedding, prediction])

    x = Dense(256, activation='relu', kernel_regularizer=l2(reg_lambda), name='dense_256')(concatenated)
    x = Dense(128, activation='relu', kernel_regularizer=l2(reg_lambda), name='dense_128')(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(reg_lambda), name='dense_32')(x)
    output = Dense(1, activation='linear', name='output')(x)

    model = Model(inputs=[user_input, item_input, genre_input], outputs=output, name='NCF_MF_with_Bias')

    model.compile(optimizer='adam', loss='mse')

    return model


# # Hybrid rounding function
# def hybrid_rounding(predicted_matrix, threshold=0.15):
#     adjusted_matrix = np.copy(predicted_matrix)
    
#     for user in range(predicted_matrix.shape[0]):
#         for item in range(predicted_matrix.shape[1]):
#             pred = predicted_matrix[user, item]
#             # Check if prediction is within the threshold of an integer value
#             if abs(pred - np.round(pred)) <= threshold:
#                 adjusted_matrix[user, item] = np.round(pred)  # Round to the nearest integer
#             else:
#                 adjusted_matrix[user, item] = pred  # Keep the original prediction
    
#     return adjusted_matrix

# Function to normalize the ratings
'''def normalize_ratings(ratings):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_ratings = scaler.fit_transform(ratings.reshape(-1, 1)).flatten()
    return normalized_ratings, scaler'''


# # Prepare missing data to predict
# def prepare_missing_train_data(train_data, movie_genre_encoding):
#     missing_user_item_pairs = []
#     genre_inputs = []

#     for user_id in range(train_data.shape[0]):
#         for item_id in range(train_data.shape[1]):
#             if train_data[user_id, item_id] == 0:
#                 missing_user_item_pairs.append([user_id, item_id])
#                 genre_inputs.append(movie_genre_encoding[item_id])

#     missing_user_item_pairs = np.array(missing_user_item_pairs)
#     genre_inputs = np.array(genre_inputs)

#     return missing_user_item_pairs, genre_inputs


if __name__ == '__main__':
    # Argument parser for file input
    parser = argparse.ArgumentParser(description='Generate a completed ratings table.')
    parser.add_argument("--name", type=str, default="ratings_eval.npy",
                        help="Name of the npy of the ratings table to complete")
    args = parser.parse_args()

    # Load Ratings table
    print('Loading ratings...')
    table = np.load(args.name)  # DO NOT CHANGE THIS LINE
    print('Ratings loaded.')

    # Any method for matrix completion goes here
    train_data = table.copy()

    # Assume the genres and names are in another file
    # movie_genre_encoding, genre_list = prepare_genres_encoding(namesngenre)  # Add genre preprocessing here

    
    num_users = train_data.shape[0] 
    num_items = train_data.shape[1]
    embedding_dim = 32
    num_genres = len(genre_list)

    user_ids_train, item_ids_train, ratings_train = [], [], []
    for user in range(num_users):
        for item in range(num_items):
            if not np.isnan(train_data[user, item]):
                user_ids_train.append(user)
                item_ids_train.append(item)
                ratings_train.append(train_data[user, item])

    user_ids_train = np.array(user_ids_train)
    item_ids_train = np.array(item_ids_train)
    ratings_train = np.array(ratings_train)

    # Calculate global average rating
    global_average = np.mean(ratings_train)

    
    # Build and train the model
    model = build_ncf_model_genres_with_mf_and_bias(num_users, num_items, num_genres, embedding_dim, global_average, reg_lambda=1e-5)
    model.fit([user_ids_train, item_ids_train, movie_genre_encoding[item_ids_train]], ratings_train, epochs=5, batch_size=256)
    
    # Make predictions for the entire matrix
    user_ids_full = np.repeat(np.arange(num_users), num_items)
    item_ids_full = np.tile(np.arange(num_items), num_users)
    genres_full = np.tile(movie_genre_encoding, (num_users, 1))

    predictions = model.predict([user_ids_full, item_ids_full, genres_full])
    predicted_matrix = predictions.reshape((num_users, num_items))

 # Apply hybrid rounding
    # predicted_matrix = hybrid_rounding(predicted_matrix, threshold=0.15)

    table=predicted_matrix.copy()
    # Save the completed ratings table
    np.save("output.npy", table)  # DO NOT CHANGE THIS LINE
    print('Completed ratings saved to output.npy.')