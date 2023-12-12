#!/usr/bin/env python
# coding: utf-8

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
# from sklearn.decomposition import PCA
# import seaborn as sns

# Load Data, keep 6 categories only, for the csv, only keep nummerical data

df = pd.read_csv("clean_data_processed_final.csv")
number_of_categories = df['category'].value_counts().size
list_of_categories = df['category'].value_counts().index
print(number_of_categories, list_of_categories)
print(df.shape)


# Create a new column named 'record_id' to index the dataframe
df['record_id'] = range(len(df))
print(df['record_id'].nunique())


# ## Drop non-nummerical columns <- Need to explain the reasoning specifically when using Kmeans
product = df.drop(columns=['subcategory', 'name', 'brand', 'brand_url', 'codCountry',
                           'variation_0_color', 'variation_0_image',
                           'variation_1_color',
                           'variation_1_image', 'url', 'image_url', 'model','id', 'image_variation',
                           'color_variation'])
print(product.describe())

print(product.head())

# Keep category columns, only use the nummerical columns into models (hence save them into columnn_names variable)
column_names = ['current_price','raw_price','discount', 'likes_count']
print(product[column_names].corr())

# ### Aply WSS for the data
# ## calculate WSS, decided to have k = 3 due to the elbow method

wss_list = []
# Calculate the WSS for different values of k
for k in range(1, 11):
    km = KMeans(n_clusters=k)
    km.fit(product[column_names])
    wss = km.inertia_
    wss_list.append(wss)
    # print("k = {} : WSS = {}".format(k, wss))

# Create a dataframe
df1 = pd.DataFrame({"k": range(1, 11), "wss": wss_list})

# Plot the WSS vs k
plt.plot(df1["k"], df1["wss"],marker='o', linestyle='--', color='g')
plt.xlabel("Number of Clusters")
plt.ylabel("Within Sum of Squares")
plt.title("Elbow Method for Optimal k")
plt.show()


# ## Keep the category collumn, only fit the rest to the model, scale the data using standardScaler
# Find the highest value in raw_price
max_price = product['raw_price'].max()

# Find the highest value in likes_count
max_likes = product['likes_count'].max()

# Display the results
print("Highest value in raw price:", max_price)
print("Highest value in likes_count:", max_likes)
# create a new df result for not messing up data
result = product.copy(deep=True)


# The highest raw price is 544 and the highest likes_count is 21000.
# This means that the likes_count feature has a much larger scale than the price feature. Decided to scale the data.
# Create a pipeline with scaling and KMeans clustering
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kmeans', KMeans(n_clusters=3, n_init='auto', init='k-means++',
                      algorithm='lloyd',random_state=75)) # fine tuning model
])

# Apply the pipeline
scaled_product = pipeline.named_steps['scaler'].fit_transform(product[column_names])
result['cluster'] = pipeline.named_steps['kmeans'].fit_predict(scaled_product)

# ### Print out the product means, we can see that the most samples coming 
# from the cheapest prices with the least like counts (which can be understandable that cheap items can be affordable, 
# not the popular one), the best is cluster 1 with most likes_count and ~51$ raw prices.

product_means = result.groupby('cluster').agg({'current_price':'mean',
                                                'raw_price':'mean',
                                                'discount':'mean',
                                                'likes_count':'mean',
                                                'cluster':'count'})
print(product_means)    
# cluster_describe = result.groupby('cluster')[['likes_count','current_price','raw_price','discount']].describe()
# transposed_describe = cluster_describe.T
# print(transposed_describe)

# ## Prepare for Visualisation

scaled_product_df = pd.DataFrame(scaled_product, columns=column_names)
print(scaled_product_df.head())

# #### Direct features comparision: there is not so well-informed as we can see current price, discount and raw price correlate together, so we decided to apply tSNE
fig, axs = plt.subplots(3, 2, figsize=(20,20))
columns = column_names
j2, i2 = 0, 0
for i in range(len(columns)-1):
     for j in range(i+1,len(columns)):
        if j2 > 1:
            j2 = 0
            i2 += 1
        axs[i2,j2].scatter(scaled_product_df[columns[i]],
scaled_product_df[columns[j]], c=result['cluster'])
        axs[i2,j2].set_title('{} vs {}'.format(columns[i],
    columns[j]))
        j2 += 1


# ## Visualisation: Using tsne (may take upto 7 mins to run the tsne)
# Apply T-SNE with 2 components to reduce the dimensionality of the dataset
tsne = TSNE(n_components=2, random_state=27)
X_tsne = tsne.fit_transform(scaled_product)

# X_tsne
# ### We can see that the cluster are not well-separated, and non linear. recommend to use non linear algo of classification to explore more

# Create a scatter plot of the T-SNE results
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=result['cluster'], cmap='viridis')
plt.colorbar(ticks=range(len(result['cluster'] )), label='Cluster')
plt.show()

# Print out cluster column as label
# result
# ## the 10 best product with most likes_count from cluster 1 (the best cluster)

# ## Men appeared to be the best category with the most representation among the samples of the best cluster

# Filter the DataFrame for products in cluster 1 - the best cluster
best_cluster_df = result[result['cluster'] == 1]


def get_top_10(df):
    for category in df['category'].unique():
        top_10_products = df[df['category'] == category].sort_values(by=['likes_count'], ascending=False)[:10]
        print(top_10_products)


get_top_10(best_cluster_df)

# best category is the most representative in the best cluster
best_category = best_cluster_df['category'].value_counts().idxmax()
print("\n")
print(f"The best category with the most represetation in the best cluster is: {best_category}")

## Prepare for classification group with the label coming from kmeans
## merge the result with the original data frame

columns_to_show = ['category_x',	'current_price_x',
                   'raw_price_x',	'discount_x','likes_count_x',	'record_id',
                   'cluster','subcategory', 'name', 'brand', 'brand_url', 'codCountry',
                  'variation_0_color', 'variation_0_image',
                  'variation_1_color','variation_1_image', 'url', 'image_url',
                    'model','id', 'image_variation','color_variation']
merged_df = result.merge(df, on='record_id',how='left', left_index=False, right_index=False)
final_df = merged_df[columns_to_show]
print(final_df.head())

#export label for classification task
final_df.to_csv("output_cluster_kmeansv3.csv")
