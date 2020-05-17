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

PATH = "data"

class StoreValue(object):

  def __init__(self):
    self._user_id = []
    self._isbn = []

  @property
  def user_id(self):
    return self._user_id

  @user_id.setter
  def user_id(self, value):
    self._user_id = value

  @property
  def isbn(self):
    return self._isbn

  @isbn.setter
  def isbn(self, value):
    self._isbn = value


def get_data():
    with zipfile.ZipFile("data.zip") as archive:
        return (
        csv.DictReader(
            (x.decode("utf-8", "ignore") for x in archive.open(os.path.join(PATH, "ratings.csv"))), delimiter=",",
        ),
        csv.DictReader(
            (x.decode("utf-8", "ignore") for x in archive.open(os.path.join(PATH,"books.csv"))), delimiter=","
        ),
        # csv.DictReader(
        #     (x.decode("utf-8", "ignore") for x in archive.open(os.path.join(PATH,"BX-Users.csv"))), delimiter=";"
        # ),
        )


#fetch the interactions data
def get_ratings():
    return get_data()[0]

#fetch the item features
def get_book_features():
    return get_data()[1]

#fetch the user features
# def get_user_features():
#     return get_data()[2]

k=0
id_isbn = StoreValue()
# for x in get_ratings():
#     print(x)
# print(get_data()['book_id'][0])
for x in get_ratings():
    if k==5000:
        break
    id_isbn._user_id.append(x['user_id'])
    id_isbn._isbn.append(x['book_id'])
    k+=1
    
# print(id_isbn._user_id)
dataset = Dataset()
dataset.fit(id_isbn._user_id,
            id_isbn._isbn)

num_users, num_items = dataset.interactions_shape()
print('Num users: {}, num_items {}.'.format(num_users, num_items))

dataset.fit_partial(items=(x['book_id'] for x in get_book_features()),
                    item_features=(x['authors'] for x in get_book_features()))


(interactions, weights) = dataset.build_interactions((id_isbn._user_id[i], id_isbn._isbn[i]) for i in range(5000))

item_features = dataset.build_item_features(((x['book_id'], [x['authors']])
                                              for x in get_book_features()))

print(item_features.shape)
print(interactions.shape)
# print(weights)

#################################
#								#
#  		Training the Model 		#
#								#
#################################

# print(interactions.tocsr()[0])
model = LightFM_ext(loss='warp')

# print("========================================")
# print(model)
(train, test) = random_train_test_split(interactions=interactions, test_percentage=0.02)
# print(test.shape)
model.fit(train, item_features=item_features, epochs=2, num_threads=4)
# print(model)
print("======================================>")
# test_precision = auc_score(model, test, item_features=item_features).mean()
# print(test_precision)
# train_precision = precision_at_k(model, train,item_features=item_features, user_features=user_features, k=2).mean()
# test_precision = precision_at_k(model, test, item_features=item_features, user_features=user_features,k=2).mean()

# train_auc = auc_score(model, train,item_features=item_features, user_features=user_features).mean()
# test_auc = auc_score(model, test,item_features=item_features, user_features=user_features).mean()

# print('Precision: train %.2f, test %.2f.' % (train_precision, test_precision))
# print('AUC: train %.2f, test %.2f.' % (train_auc, test_auc))

labels = np.array([x['book_id'] for x in get_ratings()])

def sample_recommendation(model, data, user_ids):

    n_users, n_items = data.shape

    #build a structure to store user scores for each item
    all_scores = np.empty(shape=(0,n_items))

    #iterate through the group and build the scores
    for user_id in user_ids:
        #known_positives = labels[data.tocsr()[user_id].indices]

        scores = model.predict(user_id,np.arange(n_items),item_features)
        print(scores)
        top_items_for_user = labels[np.argsort(-scores)]
        print("Top Recommended book_id For User: ", user_id)
        for x in top_items_for_user[:3]:
            print("     %s" % x)

        #vertically stack the user scores (items are columns)
        all_scores = np.vstack((all_scores, scores))
        #print(all_top_items)

    #compute the average rating for each item in the group
    item_averages = np.mean(all_scores.astype(np.float), axis=0)
    top_items_for_group = labels[np.argsort(-item_averages)]

    print("Top Recommended book_id for Group:")

    for x in top_items_for_group[:5]:
        print("     %s" % x)

with open('model_lfe-10-components-200-epoch.pkl', 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('explicit_interactions.pkl', 'wb') as f:
    pickle.dump(interactions, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('explicit_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('item_features.pkl', 'wb') as f:
    pickle.dump(item_features, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('weights.pkl', 'wb') as f:
    pickle.dump(weights, f, protocol=pickle.HIGHEST_PROTOCOL)
# with open('weight.pickle', 'wb') as f:
#     pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('model_lfe-10-components-200-epoch.pkl', 'rb') as f:
    model_x = pickle.load(f)

print(model_x.toarray())
# #fetch user_ids of users in group
# group = [3,26,451,23,24,25]
# # print(weights)
# #sample recommendations for the group
# sample_recommendation(model_x, interactions, group)
