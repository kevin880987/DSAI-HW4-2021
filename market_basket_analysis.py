#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import os

root = os.getcwd() + os.sep + 'data' + os.sep


#### Data


orders = pd.read_csv(root + 'orders.csv')
order_products_prior = pd.read_csv(root + 'order_products__prior.csv')
order_products_train = pd.read_csv(root + 'order_products__train.csv')
products = pd.read_csv(root + 'products.csv')

order_products = order_products_prior.append(order_products_train)

# Out of 49685 keeping top 100 most frequent products.
product_counts = order_products.groupby('product_id')['order_id'].count().reset_index().rename(columns = {'order_id':'frequency'})
product_counts = product_counts.sort_values('frequency', ascending=False)[0:100].reset_index(drop = True)
product_counts = product_counts.merge(products, on = 'product_id', how = 'left')

# Keeping 100 most frequent items in order_products dataframe
freq_products = list(product_counts.product_id)
order_products = order_products[order_products.product_id.isin(freq_products)]
order_products = order_products.merge(products, on = 'product_id', how='left')

# Structuring the data for feeding in the algorithm
basket = order_products.groupby(['order_id', 'product_name'])['reordered'].count().unstack().reset_index().fillna(0).set_index('order_id')

del product_counts, products, order_products, order_products_prior, order_products_train

# encoding the units
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1 
    
basket = basket.applymap(encode_units)

# Creating frequent sets and rules
frequent_items = apriori(basket, min_support=0.01, use_colnames=True, low_memory=True)
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules.sort_values('lift', ascending=False, inplace=True)
rules[['antecedents', 'consequents']] = rules[['antecedents', 'consequents']].applymap(lambda x: list(x)[0])

rules.to_csv(root+'association_rules.csv')

