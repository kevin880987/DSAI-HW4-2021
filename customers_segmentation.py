#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
pd.set_option('max_columns', 150)

import gc
import os

# matplotlib and seaborn for plotting
import matplotlib
matplotlib.rcParams['figure.dpi'] = 144 #resolution
matplotlib.rcParams['figure.figsize'] = (8,6) #figure size

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import seaborn as sns
# sns.set_style('white')
color = sns.color_palette()

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

root = os.getcwd() + os.sep + 'data' + os.sep
image_fp = os.getcwd() + os.sep + 'image' + os.sep


#### Data
aisles = pd.read_csv(root + 'aisles.csv')
departments = pd.read_csv(root + 'departments.csv')
orders = pd.read_csv(root + 'orders.csv')
order_products_prior = pd.read_csv(root + 'order_products__prior.csv')
order_products_train = pd.read_csv(root + 'order_products__train.csv')
products = pd.read_csv(root + 'products.csv')

# For segmentation, consider users from prior set only
order_products = order_products_prior.merge(products, on ='product_id', how='left')
order_products = order_products.merge(aisles, on ='aisle_id', how='left')
order_products = order_products.merge(departments, on ='department_id', how='left')
order_products = order_products.merge(orders, on='order_id', how='left')

#### Segmentation
# Since there are thousands of products in the dataset, we rely on aisles, 
# which represent categories of products. 
# Use Principal Component Analysis to find new dimensions along which clustering will be easier.
cross_df = pd.crosstab(order_products.user_id, order_products.aisle)

# Normalize each row
df = cross_df.div(cross_df.sum(axis=1), axis=0)

##### PCA and K-Means Clustering
# Reducing this dataframe to only 10 dimensions as KMeans does not work properly in higher dimension. 
pca = PCA(n_components=10)
df_pca = pca.fit_transform(df)
df_pca = pd.DataFrame(df_pca)

Sum_of_squared_distances = []
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_pca)
    Sum_of_squared_distances.append(km.inertia_)

plt.subplots(figsize = (8, 5))
plt.plot(K, Sum_of_squared_distances, '.-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method for Optimal k')
plt.gcf().tight_layout()
plt.gcf().savefig(image_fp+'Optimal k.png', dpi=144, transparent=True)
plt.show()


# From above plot we can choose optimal K as 5
k = 5 # 6 # 
clusterer = KMeans(n_clusters=k,random_state=42).fit(df_pca)
centers = clusterer.cluster_centers_
c_preds = clusterer.predict(df_pca)

# # Visualizing clustering among first two principal components
# temp_df = df_pca.iloc[:, 0:2]
# temp_df.columns = ["pc1", "pc2"]
# temp_df['cluster'] = c_preds

# fig, ax = plt.subplots(figsize = (8, 5))
# ax = sns.scatterplot(data = temp_df, x = "pc1", y = "pc2", hue = "cluster")
# ax.set_xlabel("Principal Component 1")
# ax.set_ylabel("Principal Component 2")
# ax.set_title("Cluster Visualization")
# plt.show()


#### Top products per cluster
cross_df['cluster'] = c_preds

# Customer Segmentation Results:
fig, ax = plt.subplots(figsize = (4,3))
ax = sns.countplot(cross_df['cluster'], color = color[0])
ax.set_xlabel('Cluster', size = 10)
ax.set_ylabel('Number of Users', size = 10)
ax.tick_params(axis = 'both', labelsize = 8)
ax.set_title('Total Users of each Cluster')
fig.tight_layout()
fig.savefig(image_fp+'Total Users of each Cluster.png', transparent=True)
plt.show()

# cluster1 = cross_df[cross_df.cluster == 0]
# cluster2 = cross_df[cross_df.cluster == 1]
# cluster3 = cross_df[cross_df.cluster == 2]
# cluster4 = cross_df[cross_df.cluster == 3]
# cluster5 = cross_df[cross_df.cluster == 4]

# cluster1.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]
# cluster2.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]
# cluster3.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]
# cluster4.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]
# cluster5.drop('cluster',axis=1).mean().sort_values(ascending=False)[0:10]

# - Cluster 1 results into 5428 consumers having a very strong preference for water seltzer sparkling water aisle.
# - Cluster 2 results into 55784 consumers who mostly order fresh vegetables followed by fruits.
# - Cluster 3 results into 7948 consumers who buy packaged produce and fresh fruits mostly.
# - Cluster 4 results into 37949 consumers who have a very strong preference for fruits followed by fresh vegetables.
# - Cluster 5 results into 99100 consumers who orders products from many aisles. Their mean orders are low compared to other clusters which tells us that either they are not frequent users of Instacart or they are new users and do not have many orders yet. 

# Encode the labels and save
import category_encoders as ce
encoder= ce.BinaryEncoder(cols=['cluster'], return_df=True)
clus_feats = encoder.fit_transform(cross_df['cluster'])
clus_feats.isnull().any().any()

clus_feats.to_pickle(root + 'cluster.pkl')




