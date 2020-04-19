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
user_id = 183

# Đọc data sách, trong này chứa thông tin của những cuốn sách, file này trong dataset goodbooks-10k
books = pd.read_csv('model/books.csv')

# This is an edited/extended/custom LightFM model
# 10 components 200 epochs
# Load model lên
with open('model/model_lfe-10-components-200-epoch.pkl', 'rb') as f:
    model_explicit_ratings = pickle.load(f)

# Được tạo ra theo hướng dẫn tại https://making.lyst.com/lightfm/docs/examples/dataset.html với số users: 53425 và số items 10000.
with open('model/explicit_dataset.pkl', 'rb') as f:
    dataset_explicit = pickle.load(f)

# interactions: dưới dạng COO_maxtrix, các tương tác sẽ là User-ID và ISBN của sách được cung cấp trong BX-Book-Ratings.csv
with open('model/explicit_interactions.pkl', 'rb') as f:
    interactions_explicit = pickle.load(f)

# Trọng số sau khi training mô hình
with open('model/weights.pkl', 'rb') as f:
    weights_explicit = pickle.load(f)

# This is the full item_features matrix
# Đây là đặc trưng trích xuất từ các items (sách) dựa trên tác giả của cuốn sách được cung cấp trong BX-Books.csv
with open('model/item_features.pkl', 'rb') as f:
    item_features = pickle.load(f)

@app.route('/')
def explicit_ratings():
    return render_template('book-recommender-explicit-ratings.html')

@app.route('/explicit-recommendations-ratings', methods = ['GET','POST'])
def explicit_recs_ratings():
    if request.method == 'GET':
        # print(interactions_explicit)
        predictions = predict_for_user_explicit_lightfm(model_explicit_ratings, dataset_explicit, interactions_explicit,
        books, item_features=item_features, model_user_id = user_id, num_recs = 24).to_dict(orient='records')

        return render_template('explicit_recommendations_ratings.html', predictions = predictions)
    
    if request.method == 'POST':
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
        books, item_features=item_features, model_user_id = user_id, num_recs = 24).to_dict(orient='records')
        time_elapsed = time.time() - start_time
        print('Predictions generated')
        print(f'Time to run: {datetime.timedelta(seconds=time_elapsed)}')
        sys.stdout.flush()
        item_id_map = dataset_explicit.mapping()[2]
        all_item_ids = sorted(list(item_id_map.values()))
        predicted = model_explicit_ratings.predict(user_id, all_item_ids)
        actual = weights_explicit_arr[user_id]
        nonzero_actual = np.nonzero(actual)
        sort_inds = predicted[nonzero_actual].argsort()[::-1]
        r = actual[nonzero_actual][sort_inds]
        ndcg = ndcg_at_k(r, 5)
        print(f'nDCG for current user: {ndcg}')

        return render_template('explicit_recommendations_ratings.html', predictions = predictions)

if __name__ == "__main__":
    app.run()