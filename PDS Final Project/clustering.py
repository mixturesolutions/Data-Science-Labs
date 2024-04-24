import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# Load reduced-dimensional features from Features_Reduced.pt
features_reduced = torch.load('Features_Reduced.pt')

# Perform hierarchical clustering using the Ward method
linked = linkage(features_reduced, method='ward')

# Number of clusters
n_clusters = 5  # Adjust the number of clusters as needed

# Form flat clusters from the hierarchical clustering tree
cluster_labels = fcluster(linked, n_clusters, criterion='maxclust')

# Optionally visualize the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.show()

# Define the filename for the exported cluster labels
output_file = 'Cluster_Labels.txt'

# Export cluster labels to a text file
np.savetxt(output_file, cluster_labels, fmt='%d')

print(f"Cluster labels exported to {output_file}")

# Saving the linkage matrix for later use
torch.save(linked, 'Hierarchical_model.pth')