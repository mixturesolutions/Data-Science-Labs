import torch
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# Load reduced-dimensional features of the training set
training_features_reduced = torch.load('Features_Reduced.pt')

# Optionally, select a subset of the training features to speed up training
subset_size = int(len(training_features_reduced))
subset_indices = np.random.choice(len(training_features_reduced), size=subset_size, replace=False)
training_features_subset = training_features_reduced[subset_indices]

# Extract ground truth labels from the training data
csv_path = '/Users/mixturesolution/Desktop/Data-Science-Labs/PDS Final Project/birds.csv'
df = pd.read_csv(csv_path, header=None)
train_data = df[df[3] == 'train']
ground_truth_labels = train_data[4].tolist()

# Get the number of unique species labels
species_labels = train_data[4].unique()
n_species = len(species_labels)

# Instantiate KMeans with the number of unique species labels as the number of clusters
kmeans = KMeans(n_clusters=n_species)

# Fit KMeans to the training features
kmeans.fit(training_features_subset)

# Get cluster labels for each sample
cluster_labels = kmeans.labels_

# Define the filename for the trained model
model_filename = 'KMeans_model.pth'

# Save the trained model
torch.save(kmeans, model_filename)

# Map cluster labels to species labels
predicted_species_labels = [species_labels[label] for label in cluster_labels]

# Calculate accuracy
accuracy = accuracy_score(ground_truth_labels, predicted_species_labels)
print(f"Accuracy on the training set: {accuracy}")