#!pip install kneed
#use the above if not already installed
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("clean_data_processed.csv")
df.shape
df.head()

print(df['url'].value_counts())
print(df['image_url'].value_counts())

print(df.loc[df['image_url'] == 0])

df2 = df.drop(columns=['image_url', 'url', 'variation_0_color', 'variation_1_color',
                       'variation_0_image', 'variation_1_image', 'Unnamed: 0'])
#print(df2)

df3 = df2.drop(columns=['name', 'subcategory', 'brand',
                        'brand_url', 'id', 'model', 'codCountry'])
#print(df3)

def onehotencode(dataframe, feature):
    dummies = pd.get_dummies(dataframe[[feature]])
    new_df = pd.concat([dataframe, dummies], axis=1)
    new_df = new_df.drop([feature], axis=1)
    return(new_df)

df4 = df3
df4 = onehotencode(df4, 'category')
print("amount of columns after one hot encode = "+ str(len(df4.columns)))

scaler_df4 = StandardScaler()
df4_scaled = scaler_df4.fit_transform(df4.to_numpy())
df4 = pd.DataFrame(df4_scaled, columns=df4.columns)

#print(df4)
def neighbour_analysis(dfn, k):

    if 'label' in dfn:
        dfn.drop(columns=['label'])
    nbrs = NearestNeighbors(n_neighbors = k).fit(dfn)
    # Find the k-neighbors of a point
    neigh_dist, neigh_ind = nbrs.kneighbors(dfn)
    # sort the neighbor distances (lengths to points) in ascending order
    # axis = 0 represents sort along first axis i.e. sort along row
    sort_neigh_dist = np.sort(neigh_dist, axis=0)
    #closest_distances = sort_neigh_dist[:,1]
    furthest_distances = sort_neigh_dist[:,k-1]
    #plt.plot(closest_distances)
    plt.plot(furthest_distances)
    plt.ylim(0,20)
    plt.ylabel("Eps candidate values")
    plt.xlabel("distance to kth NN")
    plt.show()

    kneedle = KneeLocator(x = range(1, len(neigh_dist)+1),
                          y = furthest_distances, S = 1.0,
                          curve = "concave",
                          direction = "increasing", online=True)
    print(kneedle.knee_y)

def cluster_count_across_ms(dfn, ms_range, sample_range):
    for ms in ms_range:
        for sample in sample_range:
            if 'label' in dfn:
                df5.drop(columns=['label'])
            nbrs = NearestNeighbors(n_neighbors = ms).fit(dfn)

            neigh_dist, neigh_ind = nbrs.kneighbors(dfn)

            sort_neigh_dist = np.sort(neigh_dist, axis=0)

            furthest_distances = sort_neigh_dist[:,ms-1]

            kneedle = KneeLocator(x = range(1, len(neigh_dist)+1),
                          y = furthest_distances, S = 1.0,
                          curve = "concave",
                          direction = "increasing", online=True)
            e = kneedle.knee_y

            if 'label' in dfn:
                dfn.drop(columns=['label'])
            dfnb = dfn.sample(frac=sample, replace=True, random_state=2)
            cluster_label_final = DBSCAN(eps=e, min_samples=ms).fit_predict(dfnb)
            dfnb['label'] = cluster_label_final

            print("ms = " + str(ms) + ", sample = " + str(sample))
            print("e = " + str(e))
            print("cluster count = " + str(max(dfnb['label'].unique())+1))
            print()

neighbour_analysis(df4, 15)
cluster_count_across_ms(df4, range(14, 16+1), [round(0.1*i, 1) for i in range(1,11)])

def sample_cluster_display_df4(sample, e, ms):
    if 'label' in df4:
        df4.drop(columns=['label'])
    df4b = df4.sample(frac=sample, replace=True, random_state=2)
    df4b = df4b.sort_index()
    cluster_label_final = DBSCAN(eps=e, min_samples=ms).fit_predict(df4b)
    if sample == 1:
        df4b_scaled = scaler_df4.inverse_transform(df4b.to_numpy())
        df4b_scaled = pd.DataFrame(df4b_scaled, columns=df4b.columns)
        df4b = df4b_scaled
    df4b['label'] = cluster_label_final
    columns = list(df4b.columns)
    columns.remove('label')
    columns = columns[:4]
    print("cluster count = " + str(max(df4b['label'].unique())+1))
    print(df4b['label'].value_counts())
    #print(columns)
    for i in range(len(columns)-1):
      for j in range(i+1,len(columns)):
        legend_list=[]
        for k in range(-1, max(df4b['label'].unique())+1):
          legend_list.append(k)
          df_temp = df4b[df4b['label']==k]
          if k>=7:
            k += 1
          if k==-1:
            k = 7
          plt.scatter(df_temp[columns[i]], df_temp[columns[j]], c="C"+str(k))
        plt.title("{} vs {}".format(columns[i], columns[j]))
        plt.legend(legend_list)
        plt.show()
    df4b_mean = df4b.groupby(['label']).agg('mean')
    print(df4b_mean[columns])
    print(df4b_mean[["image_variation","color_variation"]])

sample_cluster_display_df4(1, 14.458020743340592, 15)

df5 = df3.drop(columns=['category', 'color_variation', 'image_variation'])
#print(df5.columns) #current_price, raw_price, discount, likes_count
#print(df5)

scaler_df5 = StandardScaler()
df5_scaled = scaler_df5.fit_transform(df5.to_numpy())
df5 = pd.DataFrame(df5_scaled, columns=['current_price', 'raw_price', 'discount', 'likes_count'])

neighbour_analysis(df5, 5)

def sample_cluster_display_df5(sample, e, ms):
    if 'label' in df5:
        df5.drop(columns=['label'])
    df5b = df5.sample(frac=sample, replace=True, random_state=2)
    df5b = df5b.sort_index()
    cluster_label_final = DBSCAN(eps=e, min_samples=ms).fit_predict(df5b)
    if sample == 1:
        df5b_scaled = scaler_df5.inverse_transform(df5b.to_numpy())
        df5b_scaled = pd.DataFrame(df5b_scaled, columns=['current_price', 'raw_price', 'discount', 'likes_count'])
        df5b = df5b_scaled
    df5b['label'] = cluster_label_final
    columns = list(df5b.columns)
    columns.remove('label')
    print("cluster count = " + str(max(df5b['label'].unique())+1))
    print(df5b['label'].value_counts())
    print(columns)
    for i in range(len(columns)-1):
      for j in range(i+1,len(columns)):
        legend_list=[]
        for k in range(-1, max(df5b['label'].unique())+1):
          legend_list.append(k)
          df_temp = df5b[df5b['label']==k]
          if k>=7:
            k += 1
          if k==-1:
            k = 7
          plt.scatter(df_temp[columns[i]], df_temp[columns[j]], c="C"+str(k))
        plt.title("{} vs {}".format(columns[i], columns[j]))
        plt.legend(legend_list)
        plt.show()
    df5b_mean = df5b.groupby(['label']).agg('mean')
    print(df5b_mean)

#cluster_count_across_ms(df5, range(4, 6+1), [round(0.1*i, 1) for i in range(1,11)])
cluster_count_across_ms(df5, range(4, 10+1), [1])
sample_cluster_display_df5(1, 4.490907766417597, 5)

#run the code again under the new csv file
df = pd.read_csv("clean_data_processed_final.csv")
df.shape
df.head()

df2 = df.drop(columns=['image_url', 'url', 'variation_0_color', 'variation_1_color',
                       'variation_0_image', 'variation_1_image', 'Unnamed: 0'])
#print(df2)

df3 = df2.drop(columns=['name', 'subcategory', 'brand',
                        'brand_url', 'id', 'model', 'codCountry'])
#print(df3)

df4 = df3
df4 = onehotencode(df4, 'category')
print("amount of columns after one hot encode = "+ str(len(df4.columns)))

scaler_df4 = StandardScaler()
df4_scaled = scaler_df4.fit_transform(df4.to_numpy())
df4 = pd.DataFrame(df4_scaled, columns=df4.columns)

neighbour_analysis(df4, 18)
#cluster_count_across_ms(df4, range(17, 19+1), [round(0.1*i, 1) for i in range(1,11)])

sample_cluster_display_df4(1, 3.6521264145062093, 18)

df5 = df3.drop(columns=['category', 'color_variation', 'image_variation'])
#print(df5.columns) #current_price, raw_price, discount, likes_count
#print(df5)

scaler_df5 = StandardScaler()
df5_scaled = scaler_df5.fit_transform(df5.to_numpy())
df5 = pd.DataFrame(df5_scaled, columns=['current_price', 'raw_price', 'discount', 'likes_count'])

neighbour_analysis(df5, 6)
#cluster_count_across_ms(df5, range(5, 7+1), [round(0.1*i, 1) for i in range(1,11)])

sample_cluster_display_df5(1, 1.7458506243375838, 6)
