import sys
import numpy as np
import pandas as pd
import flask
import pickle
import json
import time
import datetime
from scipy import sparse
from flask import Flask, render_template, request
from lightfm import LightFM
from functions.helper_functions import (
    predict_for_user_explicit_lightfm,
    ndcg_at_k
)

from lightfm import LightFM
from scipy.sparse import coo_matrix as sp
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import zipfile
import csv
import requests
import json
from itertools import islice
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split 
import pickle
from functions.lightfm_ext import LightFM_ext
from functions.helper_functions import (
    predict_for_user_explicit_lightfm,
    ndcg_at_k
)

import mysql.connector

PATH = "data"

class StoreValue(object):

  def __init__(self):
    self._user_id = []
    self._book_id = []
    self._rating = []

  @property
  def user_id(self):
    return self._user_id

  @user_id.setter
  def user_id(self, value):
    self._user_id = value

  @property
  def book_id(self):
    return self._book_id
  
  @book_id.setter
  def book_id(self, value):
    self._book_id = value

  @property
  def rating(self):
    return self.rating
  
  @rating.setter
  def rating(self, value):
    self.rating = value

HOST_DATABASE="localhost",
USER_DATABASE="recommender",
PASSWORD_DATABASE="password"
NAME_DATABASE="book_recommender"

def get_ratings():
    mydb = mysql.connector.connect(
        host="localhost",
        user="recommender",
        passwd="password"
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("USE book_recommender")
    mycursor.execute("SELECT * FROM ratings")
    return mycursor.fetchall()

def get_books():
    mydb = mysql.connector.connect(
        host="localhost",
        user="recommender",
        passwd="password"
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("USE book_recommender")
    mycursor.execute("SELECT * FROM books")
    return mycursor.fetchall()

def get_users():
    mydb = mysql.connector.connect(
        host="localhost",
        user="recommender",
        passwd="password"
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("USE book_recommender")
    mycursor.execute("SELECT * FROM users")
    return mycursor.fetchall()


def store_ratings(user_id_s, book_id_s, rating_s):
    mydb = mysql.connector.connect(
        host="localhost",
        user="recommender",
        passwd="password",
        database="book_recommender"
    )

    mycursor = mydb.cursor()

    mycursor.execute("INSERT INTO ratings VALUES (%s, %s, %s)",  (int(user_id_s), int(book_id_s), int(rating_s)))

    mydb.commit()

    print(mycursor.rowcount, "record inserted.")


def convert_pd(books):
    books_pd = pd.DataFrame()
    _book_id = []
    _authors = []
    _title = []
    _average_rating = []
    _image_url = []
    _goodreads_book_id = []
    for i in range(len(books)):
        _book_id.append(books[i][0])
        _authors.append(books[i][7])
        _title.append(books[i][10])
        _average_rating.append(books[i][12])
        _image_url.append(books[i][21])
        _goodreads_book_id.append(books[i][1])
    books_pd['book_id'] = _book_id
    books_pd['authors'] = _authors
    books_pd['title'] = _title
    books_pd['average_rating'] = _average_rating
    books_pd['image_url'] = _image_url
    books_pd['goodreads_book_id'] = _goodreads_book_id
    return books_pd

ratings = get_ratings()
books = get_books()
users = get_users()
map_user_book = StoreValue()
for x in ratings:
    if x[0] not in map_user_book._user_id:
        map_user_book._user_id.append(x[0])
    if x[1] not in map_user_book._book_id:
        map_user_book._book_id.append(x[1])

print(map_user_book._user_id, " ", map_user_book._book_id)

def load_parameter():
    # ratings = get_ratings()
    # books = get_books()
    # users = get_users()
    books_pd = convert_pd(books)

    id_users_books = StoreValue()

    for x in ratings:
        id_users_books._user_id.append(x[0])
        id_users_books._book_id.append(x[1])
        id_users_books._rating.append(x[2])
    # print(id_users_books._user_id)
    # print(id_users_books._book_id)
    dataset_explicit = Dataset()
    dataset_explicit.fit(id_users_books._user_id,
                id_users_books._book_id)

    num_users, num_items = dataset_explicit.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    dataset_explicit.fit_partial(items=(x[0] for x in books),
                        item_features=(x[7] for x in books))
    
    dataset_explicit.fit_partial(users=(x[0] for x in users))


    #create ---> mapping
    (interactions_explicit, weights_explicit) = dataset_explicit.build_interactions((id_users_books._user_id[i], id_users_books._book_id[i], id_users_books._rating[i]) for i in range(len(ratings)))
    print(weights_explicit.shape)
    item_features = dataset_explicit.build_item_features(((x[0], [x[7]]) for x in books))
    # user_features = dataset_explicit.build_user_features(((x[0], [x[1]]) for x in users))

    model_explicit_ratings = LightFM_ext(loss='warp')
    print(interactions_explicit.shape)


    (train, test) = random_train_test_split(interactions=interactions_explicit, test_percentage=0.02)

    model_explicit_ratings.fit(train, item_features=item_features, epochs=2, num_threads=4)
    return model_explicit_ratings, dataset_explicit, interactions_explicit, weights_explicit, item_features, books_pd

model_explicit_ratings, dataset_explicit, interactions_explicit, weights_explicit, item_features, books_pd = load_parameter()
# with open('model_lfe-10-components-200-epoch.pkl', 'rb') as f:
#     model_explicit_ratings = pickle.load(f)

"""
==================================TỔNG QUAN=============================================
1. Lightfm là gì? https://www.kaggle.com/niyamatalmass/lightfm-hybrid-recommendation-system
2. Thuật toán recommend sử dụng framework lightfm tại https://github.com/lyst/lightfm
3. Các kiến thức về recommender bạn có thể tham khảo tại:
- https://towardsdatascience.com/how-to-build-a-movie-recommender-system-in-python-using-lightfm-8fa49d7cbe3b
- https://machinelearningcoban.com/2017/05/17/contentbasedrecommendersys/
- https://machinelearningcoban.com/2017/05/24/collaborativefiltering/
- https://machinelearningcoban.com/2017/05/31/matrixfactorization/
4. flask được sử dụng để tạo web server.
5. Một số kiến thức liên quan đến cách xử lý goodbooks-10k bạn có thể tham khảo tại (không giống code): https://towardsdatascience.com/build-a-machine-learning-recommender-72be2a8f96ed
6. Thực hiện theo hướng dẫn trong file README.md để chạy chương trình.
"""

app=Flask(__name__)
user_id_raw = 5
if user_id_raw not in map_user_book._user_id:
    map_user_book._user_id.append(user_id_raw)

# user_id = map_user_book._user_id.index(user_id_raw)
user_id = 4
# print(user_id)

# Đọc data sách, trong này chứa thông tin của những cuốn sách, file này trong dataset goodbooks-10k
# books = pd.read_csv('model/books.csv')
# print(books['book_id'])

# This is an edited/extended/custom LightFM model
# 10 components 200 epochs
# Load model lên
# with open('model_lfe-10-components-200-epoch.pkl', 'rb') as f:
#     model_explicit_ratings = pickle.load(f)

# # Được tạo ra theo hướng dẫn tại https://making.lyst.com/lightfm/docs/examples/dataset.html với số users: 53425 và số items 10000.
# with open('explicit_dataset.pkl', 'rb') as f:
#     dataset_explicit = pickle.load(f)

# # interactions: dưới dạng COO_maxtrix, các tương tác sẽ là User-ID và ISBN của sách được cung cấp trong BX-Book-Ratings.csv
# with open('explicit_interactions.pkl', 'rb') as f:
#     interactions_explicit = pickle.load(f)

# # Trọng số sau khi training mô hình
# with open('weights.pkl', 'rb') as f:
#     weights_explicit = pickle.load(f)

# # This is the full item_features matrix
# # Đây là đặc trưng trích xuất từ các items (sách) dựa trên tác giả của cuốn sách được cung cấp trong BX-Books.csv
# with open('item_features.pkl', 'rb') as f:
#     item_features = pickle.load(f)

@app.route('/')
def explicit_ratings():
    return render_template('book-recommender-explicit-ratings.html')

@app.route('/explicit-recommendations-ratings', methods = ['GET','POST'])
def explicit_recs_ratings():
    if request.method == 'GET':
        # print(interactions_explicit)
        predictions = predict_for_user_explicit_lightfm(model_explicit_ratings, dataset_explicit, interactions_explicit,
        books_pd, item_features=item_features, model_user_id = user_id, num_recs = 5).to_dict(orient='records')
        # print(predictions)
        return render_template('explicit_recommendations_ratings.html', predictions = predictions)
    
    if request.method == 'POST':
        start_time = time.time()
        msg = request.json['msg']
        print(msg)
        sys.stdout.flush()
        weights_array = json.loads(str(request.json['array']))

        interactions_array = np.where(np.array(weights_array)!=0, 1, 0)
        # print(interactions_array.shape)
        print('Loaded arrays')
        sys.stdout.flush()

        assert len(weights_array) == weights_explicit.shape[1]
        assert len(interactions_array) == interactions_explicit.shape[1]

        weights_explicit_arr = weights_explicit.toarray()
        interactions_explicit_arr = interactions_explicit.toarray()


        # print("mmmmmmmmmmmm", weights_array)
        # print(np.where(np.array(weights_array) > 0))
        # pos = np.where(np.array(weights_array) > 0)
        # rating_temp = np.array(weights_array)[pos]

        # book_id_raw = np.array(map_user_book._book_id)[pos]
        # for i in range(len(book_id_raw)):
        #     store_ratings(map_user_book._user_id[user_id], book_id_raw[i], rating_temp[i])

        print('Cast COO matrices as ndarray')
        sys.stdout.flush()

        weights_explicit_arr[user_id] = weights_array
        interactions_explicit_arr[user_id] = interactions_array
        print('Set last rows of ndarrays equal to weights_array/interactions_array')
        sys.stdout.flush()
        weights_explicit_aug = sparse.coo_matrix(weights_explicit_arr)
        interactions_explicit_aug = sparse.coo_matrix(interactions_explicit_arr)
        # print(weights_explicit_aug)
        # print(interactions_explicit_aug)
        print('Cast ndarrays as COO matrices')
        sys.stdout.flush()
        model_explicit_ratings.fit_partial(interactions_explicit_aug, sample_weight=weights_explicit_aug, item_features=item_features, epochs=2)
        if msg=='update':
            print('Now updating last matrix row')
            sys.stdout.flush()
            model_explicit_ratings.fit_partial_by_row(user_id, interactions_explicit_aug, sample_weight = weights_explicit_aug, item_features=item_features, epochs=50)
        else:
            print('Now fitting updated matrix to model')
            sys.stdout.flush()
            # This takes between 1.5 to 8 minutes depending on the complexity of the model
            model_explicit_ratings.fit_partial(interactions_explicit_aug, sample_weight = weights_explicit_aug, item_features=item_features, epochs=5)
        print('Model fitting complete')
        sys.stdout.flush()
        predictions = predict_for_user_explicit_lightfm(model_explicit_ratings, dataset_explicit, interactions_explicit_aug,
        books_pd, item_features=item_features, model_user_id = user_id, num_recs = 24).to_dict(orient='records')
        time_elapsed = time.time() - start_time
        print('Predictions generated')
        print(f'Time to run: {datetime.timedelta(seconds=time_elapsed)}')
        # sys.stdout.flush()
        # item_id_map = dataset_explicit.mapping()[2]
        # all_item_ids = sorted(list(item_id_map.values()))
        # predicted = model_explicit_ratings.predict(user_id, all_item_ids)
        # print(predicted)
        # actual = weights_explicit_arr[user_id]
        # nonzero_actual = np.nonzero(actual)
        # sort_inds = predicted[nonzero_actual].argsort()[::-1]
        # r = actual[nonzero_actual][sort_inds]
        # ndcg = ndcg_at_k(r, 5)
        # print(f'nDCG for current user: {ndcg}')

        return render_template('explicit_recommendations_ratings.html', predictions = predictions)

if __name__ == "__main__":
    app.run()
