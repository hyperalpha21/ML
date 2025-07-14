import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#rocket league files provided by wm coach
base_path = 'C:/Users/neeld/Downloads/'
file_names = [
    'RLtest0.csv',
    'RLtest1.csv',
    'RLtest2.csv',
    'RLtest3.csv',
    'RLtest4.csv',
    'RLtest5.csv',
    'RLtest6.csv',
    'RLtest7.csv',
    'RLtest8.csv',
    'RLtest9.csv',
]
dataframes = {}
for file_name in file_names:
    file_path = base_path + file_name
    dataframes[file_name] = pd.read_csv(file_path)
    print(f"First few rows of {file_name}:")
    print(dataframes[file_name].head(), "\n")

#avg score
for file_name, df in dataframes.items():
    avg_score = df['Score'].mean()
    print(f"Average Score in {file_name}: {avg_score}")

combined_df = pd.concat(dataframes.values(), ignore_index=True)
#see what values are missing
print(combined_df.head())

#drop NaNs
columns_to_drop = ['Centering Passes', 'Demolitions', 'Aerial Hits', 'Clearances', 'Epic Saves']
filtered_df = combined_df.drop(columns=columns_to_drop)

#df print again
print("\nFirst few rows of the updated DataFrame:")
print(filtered_df.head())

from sklearn.preprocessing import StandardScaler

#clustering variables we want to test
clustering_data = filtered_df[['Goals Scored', 'Assists', 'Saves', 'Shots']]
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)
print(clustering_data_scaled[:5])

import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#elbow method to test ideal k value
inertia = []
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_data_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.title("elbow method")
plt.xlabel("# of Clusters")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid()
plt.show()

#used k-5
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(clustering_data_scaled)
filtered_df['Cluster'] = clusters
#look at centroids
import pandas as pd
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), 
    columns=['Goals Scored', 'Assists', 'Saves', 'Shots']
)
print(cluster_centers)

sns.scatterplot(
    x=filtered_df['Goals Scored'], 
    y=filtered_df['Shots'], 
    hue=filtered_df['Cluster'], 
    palette='viridis'
)
plt.title("Player Clusters by Goals and Shots")
plt.show()

players_by_cluster = filtered_df.groupby('Cluster')['Player Name'].apply(list)
for cluster, players in players_by_cluster.items():
    print(f"Cluster {cluster}:")
    print(players)
    print("\n")

from sklearn.metrics import silhouette_score

#test cluster accuracy with silhouette score
silhouette_avg = silhouette_score(clustering_data_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.2f}")

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#testing again
k_values = range(2, 7)
inertia = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_data_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.title("Elbow Method for Optimizing Clusters")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.xticks(k_values)
plt.grid()
plt.show()

from sklearn.cluster import KMeans
import pandas as pd

#found optimal k=3
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(clustering_data_scaled)
filtered_df['Cluster'] = clusters
#look at centroids
cluster_centers = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_), 
    columns=['Goals Scored', 'Assists', 'Saves', 'Shots']
)
print(cluster_centers)

#testing dbscan
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=1.5, min_samples=2)
db_clusters = dbscan.fit_predict(clustering_data_scaled)
filtered_df['Cluster_DBSCAN'] = db_clusters
print(filtered_df.groupby('Cluster_DBSCAN').size())

numeric_columns = ['Goals Scored', 'Assists', 'Saves', 'Shots']
cluster_stats = filtered_df.groupby('Cluster_DBSCAN')[numeric_columns].mean()
print(cluster_stats)
# Calculate average stats for each cluster
cluster_stats = player_aggregated_data.groupby('Cluster')[['Goals Scored', 'Assists', 'Saves', 'Shots']].mean()
print("Cluster Stats (mean values for each cluster):")
print(cluster_stats)
# Compare individual player stats to cluster means
player_stats = player_aggregated_data[['Player Name', 'Cluster', 'Goals Scored', 'Assists', 'Saves', 'Shots']]
print(player_stats)
from sklearn.metrics import silhouette_score

#silhouette score
silhouette_avg = silhouette_score(clustering_data_scaled, kmeans.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")

#now testing agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
agg_cluster = AgglomerativeClustering(n_clusters=3)
agg_clusters = agg_cluster.fit_predict(clustering_data_scaled)
player_aggregated_data['Agg_Cluster'] = agg_clusters
print(player_aggregated_data[['Player Name', 'Agg_Cluster']])
from sklearn.metrics import silhouette_score

#agg silhouette score
silhouette_avg = silhouette_score(clustering_data_scaled, agg_cluster.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")


#weighing features to test
clustering_data_weighted = clustering_data.copy()
clustering_data_weighted['Goals Scored'] *= 2  # Emphasize Goals
clustering_data_weighted['Shots'] *= 1.5      # Emphasize Shots
clustering_data_scaled_weighted = scaler.fit_transform(clustering_data_weighted)

agg_cluster_weighted = AgglomerativeClustering(n_clusters=3)
weighted_clusters = agg_cluster_weighted.fit_predict(clustering_data_scaled_weighted)

#silhouette score again
silhouette_avg = silhouette_score(clustering_data_scaled_weighted, weighted_clusters)
print(f"Silhouette Score with weighted features: {silhouette_avg:.2f}")

#plot results
sns.pairplot(
    player_aggregated_data, 
    vars=['Goals Scored', 'Assists', 'Saves', 'Shots'], 
    hue='Agg_Cluster', 
    palette='viridis'
)
plt.savefig(r"C:\Users\neeld\Downloads\pairplot.png", dpi=300)
plt.show()

#end of V1. results seem to be pretty good, but need a lot more data to continue this. 
