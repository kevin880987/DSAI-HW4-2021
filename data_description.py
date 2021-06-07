#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
pd.set_option('max_columns', 150)

import gc
import os


import matplotlib
matplotlib.rcParams['figure.dpi'] = 144 #resolution
matplotlib.rcParams['figure.figsize'] = (8,6) #figure size
matplotlib.rcParams["font.family"] = "serif"

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

root = os.getcwd() + os.sep + 'data' + os.sep
image_fp = os.getcwd() + os.sep + 'image' + os.sep

# https://www.kaggle.com/c/instacart-market-basket-analysis/data

aisles = pd.read_csv(root + 'aisles.csv')
departments = pd.read_csv(root + 'departments.csv')
orders = pd.read_csv(root + 'orders.csv')
order_products_prior = pd.read_csv(root + 'order_products__prior.csv')
order_products_train = pd.read_csv(root + 'order_products__train.csv')
products = pd.read_csv(root + 'products.csv')


#### aisles
# This file contains different aisles and there are total 134 unique aisles.
len(aisles.aisle.unique())
aisles.aisle.unique()


#### departments


len(departments.department.unique())
departments.department.unique()


#### orders


len(orders.order_id.unique())
len(orders.user_id.unique())
orders.eval_set.value_counts()
orders.order_number.describe().apply(lambda x: format(x, '.2f'))

order_number = orders.groupby('user_id')['order_number'].max()
order_number = order_number.value_counts()

fig, ax = plt.subplots(figsize=(15,8))
ax = sns.barplot(x = order_number.index, y = order_number.values, color = color[0])
ax.set_xlabel('Orders per Customer')
ax.set_ylabel('Count')
ax.xaxis.set_tick_params(rotation=90, labelsize=10)
ax.set_title('Frequency of Total Orders by Customers')
fig.tight_layout()
fig.savefig(image_fp+'Frequency of Total Orders by Customers.png', transparent=True)

fig, ax = plt.subplots(figsize = (6,4))
ax = sns.kdeplot(orders.order_number[orders.eval_set == 'prior'], label = "Prior set", lw = 1, color=color[3])
ax = sns.kdeplot(orders.order_number[orders.eval_set == 'train'], label = "Train set", lw = 1, color=color[2])
ax = sns.kdeplot(orders.order_number[orders.eval_set == 'test'], label = "Test set", lw = 1, color=color[4])
ax.legend(["Prior Set", "Train Set", "Test Set"])
ax.set_xlabel('Number of Orders')
ax.set_ylabel('PDF')
ax.tick_params(axis = 'both', labelsize = 10)
ax.set_title('Distribution of Orders in Various Sets')
fig.tight_layout()
fig.savefig(image_fp+'Distribution of Orders in Various Sets.png', transparent=True)
plt.show()

fig, ax = plt.subplots(figsize = (4,3))
ax = sns.countplot(orders.order_dow, color=color[0])
ax.set_xlabel('Day of Week', size = 10)
ax.set_ylabel('Number of Orders', size = 10)
ax.tick_params(axis = 'both', labelsize = 8)
ax.set_title('Total Orders per Day of Week')
fig.tight_layout()
fig.savefig(image_fp+'Total Orders per Day of Week.png', transparent=True)
plt.show()

temp_df = orders.groupby('order_dow')['user_id'].nunique()
fig, ax = plt.subplots(figsize = (5,3))
ax = sns.barplot(x = temp_df.index, y = temp_df.values)
ax.set_xlabel('Day of Week', size = 10)
ax.set_ylabel('Total Unique Users', size = 10)
ax.tick_params(axis = 'both', labelsize = 8)
ax.set_title('Total Unique Users per Day of Week')
fig.tight_layout()
fig.savefig(image_fp+'Total Unique Users per Day of Week.png', transparent=True)
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
ax = sns.countplot(orders.order_hour_of_day, color = color[0])
ax.set_xlabel('Hour of Day', size = 10 )
ax.set_ylabel('Number of Orders', size = 10)
ax.tick_params(axis = 'both', labelsize = 8)
ax.set_title('Total Orders per Hour of Day')
fig.tight_layout()
fig.savefig(image_fp+'Total Orders per Hour of Day.png', transparent=True)
plt.show()

fig, ax = plt.subplots(figsize = (10,5))
ax = sns.countplot(orders.days_since_prior_order, color = color[0])
ax.set_xlabel('Days Since Prior Order', size = 10)
ax.set_ylabel('Number of Orders', size = 10)
ax.tick_params(axis = 'both', labelsize = 8)
ax.set_title('Total Orders VS Days Since Prior Order')
fig.tight_layout()
fig.savefig(image_fp+'Total Orders VS Days Since Prior Order.png', transparent=True)
plt.show()

temp_df = orders.groupby(["order_dow", "order_hour_of_day"])["order_number"].aggregate("count").reset_index()
temp_df = temp_df.pivot('order_dow', 'order_hour_of_day', 'order_number')
temp_df.head()
ax = plt.subplots(figsize=(7,3))
ax = sns.heatmap(temp_df, cmap="YlGnBu", linewidths=.5)
ax.set_title("Frequency of Day of week VS Hour of day", size = 12)
ax.set_xlabel("Hour of Day", size = 10)
ax.set_ylabel("Day of Week", size = 10)
ax.tick_params(axis = 'both', labelsize = 8)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=10)
fig = ax.get_figure()
fig.tight_layout()
fig.savefig(image_fp+"Frequency of Day of week VS Hour of day.png", transparent=True)
plt.show()


#### order_products_prior

    
len(order_products_prior.order_id.unique())
len(order_products_prior.product_id.unique())

add_to_cart_order_prior = order_products_prior.groupby('order_id')['add_to_cart_order'].count()
add_to_cart_order_prior = add_to_cart_order_prior.value_counts()
fig, ax = plt.subplots(figsize = (15,8))
ax = sns.barplot(x = add_to_cart_order_prior.index, y = add_to_cart_order_prior.values, color = color[3])
ax.set_xlabel('Number of Items in cart')
ax.set_ylabel('Count')
ax.xaxis.set_tick_params(rotation=90, labelsize = 9)
ax.set_title('Frequency of Items in Cart (Prior Set)', size = 15)
fig.tight_layout()
fig.savefig(image_fp+'Frequency of Items in Cart (Prior Set).png', transparent=True)

fig, ax = plt.subplots(figsize=(4,4))
ax = sns.barplot(x = order_products_prior.reordered.value_counts().index, 
                 y = order_products_prior.reordered.value_counts().values, color = color[3])
ax.set_xlabel('Number of Reorders', size = 10)
ax.set_ylabel('Count', size = 10)
ax.tick_params(axis = 'both', labelsize = 8)
ax.ticklabel_format(style='plain', axis='y')
ax.set_title('Frequency of Reorders (Prior Set)')
fig.tight_layout()
fig.savefig(image_fp+'Frequency of Reorders (Prior Set)', transparent=True)
plt.show()
print('Percentage of reorder in prior set:',
      format(order_products_prior[order_products_prior.reordered == 1].shape[0]*100/order_products_prior.shape[0], '.2f'))


#### order_products_train


len(order_products_train.order_id.unique())
len(order_products_train.product_id.unique())

add_to_cart_order_train = order_products_prior.groupby('order_id')['add_to_cart_order'].count()
add_to_cart_order_train = add_to_cart_order_train.value_counts()

fig, ax = plt.subplots(figsize = (15,8))
ax = sns.barplot(x = add_to_cart_order_train.index, y = add_to_cart_order_train.values, color = color[2])
ax.set_xlabel('Number of Items in cart')
ax.set_ylabel('Count')
ax.xaxis.set_tick_params(rotation=90, labelsize = 8)
# fig.tight_layout()
ax.set_title('Frequency of Items in Cart (Train Set)', size = 15)
fig.savefig(image_fp+'Frequency of Items in Cart (Train Set).png', transparent=True)

fig, ax = plt.subplots(figsize=(4,4))
ax = sns.barplot(x = order_products_train.reordered.value_counts().index, 
                y = order_products_train.reordered.value_counts().values, color = color[2])
ax.set_xlabel('Number of Reorders', size = 10)
ax.set_ylabel('Count', size = 10)
ax.tick_params(axis = 'both', labelsize = 8)
ax.set_title('Frequency of Reorder (Train Set)')
fig.tight_layout()
fig.savefig(image_fp+'Frequency of Reorder (Train Set).png', transparent=True)
plt.show()
print('Percentage of reorder in train set:',
      format(order_products_train[order_products_train.reordered == 1].shape[0]*100/order_products_train.shape[0], '.2f'))


#### products


len(products.product_name.unique())
len(products.aisle_id.unique())
len(products.department_id.unique())

temp_df = products.groupby('aisle_id')['product_id'].count()
temp_df.sort_values(ascending=False, inplace=True)
temp_df = temp_df.reset_index()

fig, ax = plt.subplots(figsize = (15,6))
ax = sns.barplot(x = temp_df.index, y = temp_df.product_id, color = color[0])
ax.set_xticklabels(temp_df.aisle_id)
ax.set_xlabel('Aisle Id')
ax.set_ylabel('Number of Products')
ax.xaxis.set_tick_params(rotation=90, labelsize = 7)
ax.set_title('Total Products per Aisle', size = 12)
fig.tight_layout()
fig.savefig(image_fp+'Total Products per Aisle.png', transparent=True)

temp_df = products.groupby('department_id')['product_id'].count()
temp_df.sort_values(ascending=False, inplace=True)
temp_df = temp_df.reset_index()
fig, ax = plt.subplots(figsize = (8,5))
ax = sns.barplot(x = temp_df.index, y = temp_df.product_id, color = color[0])
ax.set_xticklabels(temp_df.department_id)
ax.set_xlabel('Department Id')
ax.set_ylabel('Number of Products')
ax.xaxis.set_tick_params(rotation=90, labelsize = 9)
ax.set_title('Total Products per Department', size = 10)
fig.tight_layout()
fig.savefig(image_fp+'Total Products per Department.png', transparent=True)

temp_df = products.groupby('department_id')['aisle_id'].nunique()
temp_df.sort_values(ascending=False, inplace=True)
temp_df = temp_df.reset_index()

fig, ax = plt.subplots(figsize = (8,5))
ax = sns.barplot(x = temp_df.index, y = temp_df.aisle_id, color = color[0])
ax.set_xticklabels(temp_df.department_id)
ax.set_xlabel('Department Id')
ax.set_ylabel('Number of Aisles in Department')
ax.xaxis.set_tick_params(rotation=90, labelsize = 9)
ax.set_title('Total Aisles per Department', size = 10)
fig.tight_layout()
fig.savefig(image_fp+'Total Aisles in Department.png', transparent=True)




