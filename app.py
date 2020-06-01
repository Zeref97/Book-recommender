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

import sys
import numpy as np
import pandas as pd
import flask
import pickle
import json
import os
import time
import datetime
from scipy import sparse
from flask import Flask, render_template, request, flash, session, redirect, url_for
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.cross_validation import random_train_test_split 
import pickle
import mysql.connector
from functions.lightfm_ext import LightFM_ext
from scipy.sparse import coo_matrix as sp
from functions.helper_functions import (
    predict_for_user_explicit_lightfm,
    ndcg_at_k
)

"""===============================================CONFIG DATABASE========================================================"""
HOST_DATABASE = "localhost"
USER_DATABASE = "recommender"
PASSWORD_DATABASE = "password"
NAME_DATABASE = "book_recommender"


"""===============================================DEFINE STRUCT STORE VALUE========================================================"""

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


"""===============================================DATABASE OPERATION========================================================"""

# Get ratings
def get_ratings():
    mydb = mysql.connector.connect(
        host=HOST_DATABASE,
        user=USER_DATABASE,
        passwd=PASSWORD_DATABASE,
        database=NAME_DATABASE
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM ratings")
    return mycursor.fetchall()

# Get books
def get_books():
    mydb = mysql.connector.connect(
        host=HOST_DATABASE,
        user=USER_DATABASE,
        passwd=PASSWORD_DATABASE,
        database=NAME_DATABASE
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("USE book_recommender")
    mycursor.execute("SELECT * FROM books")
    return mycursor.fetchall()

# Get users
def get_users():
    mydb = mysql.connector.connect(
        host=HOST_DATABASE,
        user=USER_DATABASE,
        passwd=PASSWORD_DATABASE,
        database=NAME_DATABASE
    )
    
    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM users")
    return mycursor.fetchall()

# Store new ratings
def store_ratings(user_id_s, book_id_s, rating_s):
    mydb = mysql.connector.connect(
        host=HOST_DATABASE,
        user=USER_DATABASE,
        passwd=PASSWORD_DATABASE,
        database=NAME_DATABASE
    )

    mycursor = mydb.cursor()

    # Check book is rated? if rated --> update rate, else --> insert rate
    sql = """SELECT * FROM ratings WHERE user_id = %s AND book_id = %s """

    mycursor.execute(sql, (int(user_id_s), int(book_id_s),))
    result = mycursor.fetchall()
    if result:
        sql = """UPDATE ratings SET rating = %s WHERE user_id = %s AND book_id = %s """

        mycursor.execute(sql, (int(rating_s), int(user_id_s), int(book_id_s),))
    else:
        mycursor.execute("INSERT INTO ratings VALUES (%s, %s, %s)",  (int(user_id_s), int(book_id_s), int(rating_s)))

    mydb.commit()

    print(mycursor.rowcount, "record inserted.")

# Check login 
def query_account(user_name):
    mydb = mysql.connector.connect(
        host=HOST_DATABASE,
        user=USER_DATABASE,
        passwd=PASSWORD_DATABASE,
        database=NAME_DATABASE
    )

    mycursor = mydb.cursor(buffered=True)

    sql = """SELECT * FROM users WHERE user_name = %s"""

    mycursor.execute(sql, (user_name, ))
    return mycursor.fetchall()


"""===============================================LOAD PARAMETERS========================================================"""
# Convert book data to pandas
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

# Load and training recommender
def load_parameter():
    ratings = get_ratings()
    books = get_books()
    users = get_users()
    books_pd = convert_pd(books)

    id_users_books = StoreValue()

    for x in ratings:
        id_users_books._user_id.append(x[0])
        id_users_books._book_id.append(x[1])

    # Được tạo ra theo hướng dẫn tại https://making.lyst.com/lightfm/docs/examples/dataset.html
    dataset_explicit = Dataset()
    dataset_explicit.fit(id_users_books._user_id,
                id_users_books._book_id)

    num_users, num_items = dataset_explicit.interactions_shape()
    print('Num users: {}, num_items {}.'.format(num_users, num_items))

    dataset_explicit.fit_partial(items=(x[0] for x in books),
                        item_features=(x[7] for x in books))
    
    dataset_explicit.fit_partial(users=(x[0] for x in users))


    # create ---> mapping
    # interactions: dưới dạng COO_maxtrix, các tương tác sẽ là user_id và book_id
    # Trọng số voting
    (interactions_explicit, weights_explicit) = dataset_explicit.build_interactions((id_users_books._user_id[i], id_users_books._book_id[i]) for i in range(len(ratings)))

    # Đây là đặc trưng trích xuất từ các items (sách) dựa trên tác giả của cuốn sách được cung cấp
    item_features = dataset_explicit.build_item_features(((x[0], [x[7]]) for x in books))
    # user_features = dataset_explicit.build_user_features(((x[0], [x[1]]) for x in users))

    model_explicit_ratings = LightFM_ext(loss='warp')

    (train, test) = random_train_test_split(interactions=interactions_explicit, test_percentage=0.02)

    model_explicit_ratings.fit(train, item_features=item_features, epochs=2, num_threads=4)
    return model_explicit_ratings, dataset_explicit, interactions_explicit, weights_explicit, item_features, books_pd


"""===============================================GLOBAL VALUE========================================================"""

model_explicit_ratings, dataset_explicit, interactions_explicit, weights_explicit, item_features, books_pd = load_parameter()

app=Flask(__name__)
app.config['SESSION_TYPE'] = 'memcached'
app.config['SECRET_KEY'] = 'super secret key'
user_id = 0
name = ""

user_id_map = dataset_explicit.mapping()[0]
item_id_map = dataset_explicit.mapping()[2]

book_id_key = []
user_id_key = []

for k in item_id_map.keys():
    book_id_key.append(k)

for k in user_id_map.keys():
    user_id_key.append(k)


"""===============================================FLASK========================================================"""

#Login
@app.route('/', methods=['GET', 'POST'])
def login():
    global user_id, name

    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']

        account = query_account(username)

        if account:
            user_id_raw = account[0][0]
            for i, k in enumerate(user_id_key):
                if (k == user_id_raw):
                    user_id = i
                    break

            print("===============> user id raw %s ------ user id %s." % (user_id_raw, user_id))
                
            password = account[0][3]
            name = account[0][2]

            # Compare Passwords
            if (password == password_candidate):
                flash('You are now logged in', 'success')
                return redirect(url_for('book_recommender'))
            else:
                error = 'Invalid login'
                return render_template('login.html', error=error)

            # Close connection
            mycursor.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')

def Convert(lst): 
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)} 
    return res_dct 

@app.route('/book_recommender')
def book_recommender():
    return render_template('book-recommender-explicit-ratings.html', name=name)


# show new books
@app.route('/book_recommender/new_books', methods = ['GET','POST'])
def new_books():
    if request.method == 'GET':
        books = get_books()
        predictions = []
        for b in books[0:10]:
            predictions.append(Convert(['predictions', 0.0, 'model_book_id', [i for i, k in enumerate(book_id_key) if k == b[0]][0], 'book_id', b[0], 'authors', b[7], 
                                'title', b[10], 'average_rating', b[12], 'image_url', b[21], 'goodreads_book_id', b[1]]))
                  
        return render_template('explicit_recommendations_ratings.html', predictions = predictions)


# Generate and predict
@app.route('/explicit-recommendations-ratings', methods = ['GET','POST'])
def explicit_recs_ratings():
    # print("==================> user id: ", user_id)
    if request.method == 'GET':
        # print(request.form['search'])
        # model_explicit_ratings: train
        # dataset_explicit: books
        predictions = predict_for_user_explicit_lightfm(model_explicit_ratings, dataset_explicit, interactions_explicit,
        books_pd, item_features=item_features, model_user_id = user_id, num_recs = 5).to_dict(orient='records')
        # print(predictions)
        return render_template('explicit_recommendations_ratings.html', predictions = predictions)
    
    if request.method == 'POST':
        # print([i for (i, k) in enumerate(book_id_key) if k == 3 ][0])
        if request.json == None:
            books = get_books()
            predictions = []
            for b in books:
                if request.form['search'] in b[10]:
                    predictions.append(Convert(['predictions', 0.0, 'model_book_id', [i for i, k in enumerate(book_id_key) if k == b[0]][0], 'book_id', b[0], 'authors', b[7], 
                                        'title', b[10], 'average_rating', b[12], 'image_url', b[21], 'goodreads_book_id', b[1]]))
            # print(predictions)                         
            return render_template('explicit_recommendations_ratings.html', predictions = predictions)
        else:
            start_time = time.time()
            msg = request.json['msg']
            print(msg)
            sys.stdout.flush()
            weights_array = json.loads(str(request.json['array']))

            interactions_array = np.where(np.array(weights_array)!=0, 1, 0)

            print('Loaded arrays')
            sys.stdout.flush()

            assert len(weights_array) == weights_explicit.shape[1]
            assert len(interactions_array) == interactions_explicit.shape[1]

            weights_explicit_arr = weights_explicit.toarray()
            interactions_explicit_arr = interactions_explicit.toarray()
            print('Cast COO matrices as ndarray')
            sys.stdout.flush()
            pos = np.where(np.array(weights_array) > 0)
            rating_temp = np.array(weights_array)[pos]

            book_id_raw = np.array(book_id_key)[pos]

            for i in range(len(book_id_raw)):
                store_ratings(user_id_key[user_id], book_id_raw[i], rating_temp[i])

            weights_explicit_arr[user_id] = weights_array
            interactions_explicit_arr[user_id] = interactions_array
            print('Set last rows of ndarrays equal to weights_array/interactions_array')
            sys.stdout.flush()
            weights_explicit_aug = sparse.coo_matrix(weights_explicit_arr)
            interactions_explicit_aug = sparse.coo_matrix(interactions_explicit_arr)

            print('Cast ndarrays as COO matrices')
            sys.stdout.flush()
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
            sys.stdout.flush()
            item_id_map = dataset_explicit.mapping()[2]
            all_item_ids = sorted(list(item_id_map.values()))
            predicted = model_explicit_ratings.predict(user_id, all_item_ids)
            # print(predicted)
            actual = weights_explicit_arr[user_id]
            nonzero_actual = np.nonzero(actual)
            sort_inds = predicted[nonzero_actual].argsort()[::-1]
            r = actual[nonzero_actual][sort_inds]
            ndcg = ndcg_at_k(r, 5)
            print(f'nDCG for current user: {ndcg}')

            return render_template('explicit_recommendations_ratings.html', predictions = predictions)


"""===============================================MAIN========================================================"""

if __name__ == "__main__":
    app.run()