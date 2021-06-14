#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import os

import category_encoders as ce

root = os.getcwd() + os.sep + 'data' + os.sep


aisles = pd.read_csv(root + 'aisles.csv')

departments = pd.read_csv(root + 'departments.csv')

orders = pd.read_csv(root + 'orders.csv', 
                 dtype={
                        'order_id': np.int32,
                        'user_id': np.int64,
                        'eval_set': 'category',
                        'order_number': np.int16,
                        'order_dow': np.int8,
                        'order_hour_of_day': np.int8,
                        'days_since_prior_order': np.float32})

order_products_prior = pd.read_csv(root + 'order_products__prior.csv', 
                                 dtype={
                                        'order_id': np.int32,
                                        'product_id': np.uint16,
                                        'add_to_cart_order': np.int16,
                                        'reordered': np.int8})

order_products_train = pd.read_csv(root + 'order_products__train.csv', 
                                 dtype={
                                        'order_id': np.int32,
                                        'product_id': np.uint16,
                                        'add_to_cart_order': np.int16,
                                        'reordered': np.int8})
products = pd.read_csv(root + 'products.csv')


#### Preparing Data


prior_df = order_products_prior.merge(orders, on ='order_id', how='inner')
prior_df = prior_df.merge(products, on = 'product_id', how = 'left')
prior_df.head()


### Features creation

# Calculating how many times a user buy the product
prior_df['user_buy_product_times'] = prior_df.groupby(['user_id', 'product_id']).cumcount() + 1
prior_df.head()


#### Product level features 

# (1) Product's average add-to-cart-order
# 
# (2) Total times the product was ordered
# 
# (3) Total times the product was reordered
# 
# (4) Reorder percentage of a product
# 
# (5) Total unique users of a product
# 
## (6) Is the product Organic?
# 
# (7) Percentage of users that buy the product second time


prod_feats1 = prior_df.groupby('product_id').agg(
    mean_add_to_cart_order=('add_to_cart_order', 'mean'),
    total_orders=('reordered', 'count'), 
    total_reorders=('reordered', 'sum'), 
    reorder_percentage=('reordered', 'mean'), 
    unique_users=('user_id', lambda x: x.nunique()), 
    order_first_time_total_cnt=('user_buy_product_times', lambda x: sum(x==1)), 
    order_second_time_total_cnt=('user_buy_product_times', lambda x: sum(x==2)),
    # is_organic=('product_name', lambda x: 1 if 'Organic' in x else 0)
    )
# prod_feats1.columns = prod_feats1.columns.droplevel(0)
prod_feats1.reset_index(inplace = True)
prod_feats1.head()
prod_feats1['second_time_percent'] = prod_feats1.order_second_time_total_cnt/prod_feats1.order_first_time_total_cnt


#### Aisle and department features

# (8) Reorder percentage, Total orders and reorders of a product  aisle
# 
# (9) Mean and std of aisle add-to-cart-order
# 
# (10) Aisle unique users


aisle_feats = prior_df.groupby('aisle_id').agg(
    aisle_mean_add_to_cart_order=('add_to_cart_order', 'mean'), 
    aisle_std_add_to_cart_order=('add_to_cart_order', 'std'), 
    aisle_total_orders=('reordered', 'count'), 
    aisle_total_reorders=('reordered', 'sum'), 
    aisle_reorder_percentage=('reordered', 'mean'), 
    aisle_unique_users=('user_id', lambda x: x.nunique())
    )
# aisle_feats.columns = aisle_feats.columns.droplevel(0)
aisle_feats.reset_index(inplace = True)
aisle_feats.head()


#### **features**
# 
# (10) Reorder percentage, Total orders and reorders of a product department
# 
# (11) Mean and std of department add-to-cart-order
# 
# (12) Department unique users


dpt_feats = prior_df.groupby('department_id').agg(
    department_mean_add_to_cart_order=('add_to_cart_order', 'mean'), 
    department_std_add_to_cart_order=('add_to_cart_order', 'std'), 
    department_total_orders=('reordered', 'count'), 
    department_total_reorders=('reordered', 'sum'), 
    department_reorder_percentage=('reordered', 'mean'), 
    department_unique_users=('user_id', lambda x: x.nunique())
    )
# dpt_feats.columns = dpt_feats.columns.droplevel(0)
dpt_feats.reset_index(inplace = True)


#### **features**
# 
# (13) Binary encoding of aisle feature
# 
# (14) Binary encoding of department feature


prod_feats1 = prod_feats1.merge(products, on = 'product_id', how = 'left')
prod_feats1 = prod_feats1.merge(aisle_feats, on = 'aisle_id', how = 'left')
prod_feats1 = prod_feats1.merge(aisles, on = 'aisle_id', how = 'left')
prod_feats1 = prod_feats1.merge(dpt_feats, on = 'department_id', how = 'left')
prod_feats1 = prod_feats1.merge(departments, on = 'department_id', how = 'left')
prod_feats1.drop(['product_name', 'aisle_id', 'department_id'], axis = 1, inplace = True)

encoder= ce.BinaryEncoder(cols=['aisle', 'department'],return_df=True)
prod_feats1 = encoder.fit_transform(prod_feats1)
prod_feats1.isnull().any().any()

# free some memory
del aisle_feats, dpt_feats, aisles, departments
gc.collect()


#### User level features

# (15) User's average and std day-of-week of order
# 
# (16) User's average and std hour-of-day of order
# 
# (17) User's average and std days-since-prior-order
# 
# (18) Total orders by a user
# 
# (19) Total products user has bought
# 
# (20) Total unique products user has bought
# 
# (21) user's total reordered products
# 
# (22) User's overall reorder percentage


# when no prior order, the value is null. Imputing as 0
prior_df.days_since_prior_order = prior_df.days_since_prior_order.fillna(0)

user_feats = prior_df.groupby('user_id').agg(
    avg_dow=('order_dow', 'mean'), 
    std_dow=('order_dow', 'std'), 
    avg_doh=('order_hour_of_day', 'mean'), 
    std_doh=('order_hour_of_day', 'std'), 
    avg_since_order=('days_since_prior_order', 'mean'), 
    std_since_order=('days_since_prior_order', 'std'), 
    total_orders_by_user=('order_number', lambda x: x.nunique()), 
    total_products_by_user=('product_id', 'count'), 
    total_unique_product_by_user=('product_id', lambda x: x.nunique()), 
    total_reorders_by_user=('reordered', 'sum'), 
    reorder_proportion_by_user=('reordered', 'mean')
    )
user_feats.columns = user_feats.columns.droplevel(0)
user_feats.reset_index(inplace = True)


#### **features**
# 
# (23) Average order size of a user
# 
# (24) User's mean of reordered items of all orders

user_feats2 = prior_df.groupby(['user_id', 'order_number']).agg(
    average_order_size=('reordered', 'count'), 
    reorder_in_order=('reordered', 'mean')
    )
# user_feats2.columns = user_feats2.columns.droplevel(0)
user_feats2.reset_index(inplace = True)

user_feats3 = user_feats2.groupby('user_id').agg(
    {'average_order_size' : 'mean',
     'reorder_in_order':'mean'}
    )
user_feats3 = user_feats3.reset_index()

user_feats = user_feats.merge(user_feats3, on = 'user_id', how = 'left')


#### **features**
# 
# (25) Percentage of reordered items in user's last three orders
# 
# (26) Total orders in user's last three orders

# Last 3 orders of a user
last_three_orders = user_feats2.groupby('user_id')['order_number'].nlargest(3).reset_index()
last_three_orders = user_feats2.merge(last_three_orders, on = ['user_id', 'order_number'], how = 'inner')
last_three_orders['rank'] = last_three_orders.groupby("user_id")["order_number"].rank("dense", ascending=True)

last_order_feats = last_three_orders.pivot_table(index = 'user_id', columns = ['rank'],                                                  values=['average_order_size', 'reorder_in_order']).                                                reset_index(drop = False)
last_order_feats.columns = ['user_id','orders_3', 'orders_2', 'orders_1', 'reorder_3', 'reorder_2', 'reorder_1']

user_feats = user_feats.merge(last_order_feats, on = 'user_id', how = 'left')


#### User and Product level features

# (27) User's avg add-to-cart-order for a product
# 
# (28) User's avg days_since_prior_order for a product
# 
# (29) User's product total orders, reorders and reorders percentage
# 
# (30) User's order number when the product was bought last


user_product_feats = prior_df.groupby(['user_id', 'product_id']).agg(
    total_product_orders_by_user=('reordered', 'count'), 
    total_product_reorders_by_user=('reordered', 'sum'), 
    user_product_reorder_percentage=('reordered', 'mean'), 
    avg_add_to_cart_by_user=('add_to_cart_order', 'mean'), 
    avg_days_since_last_bought=('days_since_prior_order', 'mean'), 
    last_ordered_in=('order_number', 'max')
    )
# user_product_feats.columns = user_product_feats.columns.droplevel(0)
user_product_feats.reset_index(inplace = True)


#### **features**
# 
# (31) User's product purchase history of last three orders


last_orders = prior_df.merge(last_three_orders, on = ['user_id', 'order_number'], how = 'inner')
last_orders['rank'] = last_orders.groupby(['user_id', 'product_id'])['order_number'].rank("dense", ascending=True)

product_purchase_history = last_orders.pivot_table(index = ['user_id', 'product_id'],                                                   columns='rank', values = 'reordered').reset_index()
product_purchase_history.columns = ['user_id', 'product_id', 'is_reorder_3', 'is_reorder_2', 'is_reorder_1']
product_purchase_history.fillna(0, inplace = True)

user_product_feats = user_product_feats.merge(product_purchase_history, on=['user_id', 'product_id'], how = 'left')
user_product_feats.fillna(0, inplace = True)


#### Saving all features


prod_feats1.to_pickle(root + 'product_features.pkl')
user_feats.to_pickle(root +'user_features.pkl')
user_product_feats.to_pickle(root +'user_product_features.pkl')

# df = pd.read_pickle(root +'product_features.pkl')
# df = pd.read_pickle(root+'user_features.pkl')
# df = pd.read_pickle(root + 'user_product_features.pkl')



