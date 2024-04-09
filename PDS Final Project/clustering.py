import torch
import numpy as np
from sklearn.cluster import KMeans

# Load reduced-dimensional features from Features_Reduced.pt
features_reduced = torch.load('Features_Reduced.pt')

# Instantiate KMeans with the desired number of clusters
n_clusters = 5  # Adjust the number of clusters as needed
kmeans = KMeans(n_clusters=n_clusters)

# Fit KMeans to the reduced-dimensional features
kmeans.fit(features_reduced)

# Get cluster labels for each sample
cluster_labels = kmeans.labels_

# Define the filename for the exported cluster labels
output_file = 'Cluster_Labels.txt'

# Export cluster labels to a text file
np.savetxt(output_file, cluster_labels, fmt='%d')

print(f"Cluster labels exported to {output_file}")