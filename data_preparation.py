#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import seaborn as sns
import gc
pd.options.mode.chained_assignment = None

import os
root = os.getcwd() + os.sep + 'data' + os.sep


#### Reading all data


orders = pd.read_csv(root + 'orders.csv', 
                 dtype={
                        'order_id': np.int32,
                        'user_id': np.int64,
                        'eval_set': 'category',
                        'order_number': np.int16,
                        'order_dow': np.int8,
                        'order_hour_of_day': np.int8,
                        'days_since_prior_order': np.float32})


order_products_train = pd.read_csv(root + 'order_products__train.csv', 
                                 dtype={
                                        'order_id': np.int32,
                                        'product_id': np.uint16,
                                        'add_to_cart_order': np.int16,
                                        'reordered': np.int8})

order_products_prior = pd.read_csv(root + 'order_products__prior.csv', 
                                 dtype={
                                        'order_id': np.int32,
                                        'product_id': np.uint16,
                                        'add_to_cart_order': np.int16,
                                        'reordered': np.int8})

product_features = pd.read_pickle(root + 'product_features.pkl')
user_features = pd.read_pickle(root + 'user_features.pkl')
user_product_features = pd.read_pickle(root + 'user_product_features.pkl')
products = pd.read_csv(root +'products.csv')
aisles = pd.read_csv(root + 'aisles.csv')
departments = pd.read_csv(root + 'departments.csv')
sample_submission = pd.read_csv(root + 'sample_submission.csv')

#### Merging train order data with orders
train_orders = orders.merge(order_products_train, on = 'order_id', how = 'inner')
train_users = train_orders.user_id.unique()

#### Adding product id to test orders
test_users = orders.loc[orders.eval_set=='test'].user_id.unique()
test_prior_orders = orders.loc[orders.user_id.isin(test_users)]
test_prior_orders = test_prior_orders.merge(order_products_prior, on = 'order_id', how = 'inner')
test_prior_products = test_prior_orders.groupby('user_id').apply(lambda x: x.product_id.unique())

test_orders = orders.loc[orders.eval_set=='test']
test_orders = test_orders.merge(test_prior_products.to_frame('product_id'), on = 'user_id', how = 'inner')
test_orders = test_orders.explode('product_id')

# removing unnecessary columns from train_orders
train_orders.drop(['eval_set', 'add_to_cart_order'], axis = 1, inplace = True)
test_orders.drop(['eval_set'], axis = 1, inplace = True)

# seperate train_users, test_users data
df = user_product_features[user_product_features.user_id.isin(train_users)]
df = df.merge(train_orders, on = ['user_id', 'product_id'], how = 'outer')

test_df = user_product_features[user_product_features.user_id.isin(test_users)]
test_df = test_df.merge(test_orders, on = ['user_id', 'product_id'], how = 'outer')

# for order_number, order_dow, order_hour_of_day, days_since_prior_order, impute null values with mean values grouped by users as these products will also be potential candidate for order.
df.order_number.fillna(df.groupby('user_id')['order_number'].transform('mean'), inplace = True)
df.order_dow.fillna(df.groupby('user_id')['order_dow'].transform('mean'), inplace = True)
df.order_hour_of_day.fillna(df.groupby('user_id')['order_hour_of_day'].transform('mean'), inplace = True)
df.days_since_prior_order.fillna(df.groupby('user_id')['days_since_prior_order'].transform('mean'), inplace = True)

test_df.order_number.fillna(test_df.groupby('user_id')['order_number'].transform('mean'), inplace = True)
test_df.order_dow.fillna(test_df.groupby('user_id')['order_dow'].transform('mean'), inplace = True)
test_df.order_hour_of_day.fillna(test_df.groupby('user_id')['order_hour_of_day'].transform('mean'), inplace = True)
test_df.days_since_prior_order.fillna(test_df.groupby('user_id')['days_since_prior_order'].transform('mean'), inplace = True)

# Removing those products which were bought the first time in last order by a user
df = df[df.reordered != 0]
df.reordered.fillna(0, inplace = True)

#### Merging product and user features
df = df.merge(product_features, on = 'product_id', how = 'left')
df = df.merge(user_features, on = 'user_id', how = 'left')
# df.isnull().sum().sort_values(ascending = False)
df.set_index(['order_id', 'product_id'], inplace=True)
df.drop(['user_id'], axis = 1, inplace = True)
# df.to_csv(root + 'Finaldata.csv', index=True)
df.to_pickle(root + 'Finaldata.pkl')

test_df = test_df.merge(product_features, on = 'product_id', how = 'left')
test_df = test_df.merge(user_features, on = 'user_id', how = 'left')
# test_df.isnull().sum().sort_values(ascending = False)
test_df.set_index(['order_id', 'product_id'], inplace=True)
test_df.drop(['user_id'], axis = 1, inplace = True)
# test_df.to_csv(root + 'Testdata.csv', index=True)
test_df.to_pickle(root + 'Testdata.pkl')

