#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sys
import os


# In[9]:


script_directory = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))
print(script_directory)
if os.path.basename(script_directory).startswith("Python"):
    script_directory = "../"
    get_ipython().run_line_magic('matplotlib', 'inline')
else:
    script_directory = os.path.dirname(os.path.dirname(os.path.realpath(sys.argv[0])))


# In[10]:


df_train   = pd.read_csv(script_directory + "/data/learning/ml-learning-data.csv",   sep=';', na_values=['', '-'], parse_dates=['LastNewsletter', 'date'], dayfirst=True)
df_train['DaysSinceLastNewsletter'] = df_train['DaysSinceLastNewsletter'].str.replace(',', '.').astype(float)

df_train['id'] = [i for i in range(len(df_train.index))]
df_train.set_index('id', inplace=True)
"Training dataset: {}".format(df_train.shape)


# In[11]:


df_train.describe()


# In[12]:


df_train.dropna(inplace=True)
df_train['date'] = df_train['date'].apply(lambda x: x.timestamp())
df_train['LastNewsletter'] = df_train['LastNewsletter'].apply(lambda x: x.timestamp())


# In[13]:


def dummy_encode(in_df_train):
    df_train = in_df_train
    categorical_feats = [
        f for f in df_train.columns if df_train[f].dtype == 'object' and f != "id"
    ]
    print(categorical_feats)
    for f_ in categorical_feats:
        prefix = f_
        df_train = pd.concat([df_train, pd.get_dummies(df_train[f_], prefix=prefix)], axis=1).drop(f_, axis=1)
    return df_train


# In[14]:


df_train = dummy_encode(df_train)


# In[15]:


scaler = StandardScaler()


# In[16]:


df_train


# In[17]:


df_train[['date_t', 'DaysUntilNextPurchase_t','LastNewsletter_t', 'Units_t', 'Preis_t', 'UnitRelativeDays_t']] = scaler.fit_transform(df_train[['date', 'DaysUntilNextPurchase','LastNewsletter', 'Units', 'Preis', 'UnitRelativeDays']])


# In[18]:


df_train


# In[19]:


def optimise_k_means(data, max_k):
    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)
        
    fig = plt.subplots(figsize=(10, 5))
    plt.plot(means, inertias, 'o-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('inertia')
    plt.grid(True)
    plt.show()


# In[20]:


optimise_k_means(df_train, 10)


# In[21]:


data = df_train.values


# In[22]:


# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=3, random_state=42)
data_tsne = tsne.fit_transform(data)


# In[23]:


num_clusters = 3  # You can adjust the number of clusters based on the diagram above
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans.fit(data)


# In[24]:


labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# In[31]:


dimensions = list(df_train.columns.values)
num_dimensions = len(dimensions)
for i in range(num_dimensions):
    if dimensions[i].startswith('Customerid') \
    or dimensions[i].startswith('Orderid') \
    or dimensions[i].startswith('Produktkey'):
        continue

    for j in range(i + 1, num_dimensions):
        if dimensions[j].startswith('Customerid') \
        or dimensions[j].startswith('Orderid') \
        or dimensions[j].startswith('Produktkey'):
            continue
        x_axis = i
        y_axis = j
        
        plt.scatter(data[:, x_axis], data[:, y_axis], c=labels, cmap='viridis')
        plt.scatter(centroids[:, x_axis], centroids[:, y_axis], c='red', marker='X', label='Centroids')
        plt.legend()
        plt.xlabel(dimensions[x_axis])
        plt.ylabel(dimensions[y_axis])
        plt.savefig(f'{script_directory}/visualizations/scatter_plot_{dimensions[x_axis]}_{dimensions[y_axis]}.png')

        # Clear the current plot for the next iteration
        plt.clf()

# If you want to close all open figures at the end
plt.close('all')
# In[ ]:




