import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns# Set the environment variable to avoid the memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'# Sample dataset
data = {
'Feature1': [1.0, 1.1, 0.9, 5.0, 5.1, 4.9, 9.0, 9.1, 8.9],
'Feature2': [2.0, 2.1, 1.9, 6.0, 6.1, 5.9, 10.0, 10.1, 9.9]
}# Create DataFrame
df = pd.DataFrame(data)# Convert data to numpy array
X = df.values# K-means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Set n_init explicitly to suppress the warning
kmeans_labels = kmeans.fit_predict(X)# EM Algorithm (Gaussian Mixture Model)
gmm = GaussianMixture(n_components=3, random_state=42)  # Assuming 3 clusters
gmm_labels = gmm.fit_predict(X)# Calculate silhouette scores
kmeans_silhouette = silhouette_score(X, kmeans_labels)
gmm_silhouette = silhouette_score(X, gmm_labels)print(f"K-means Silhouette Score: {kmeans_silhouette}")
print(f"EM Silhouette Score: {gmm_silhouette}")# Visualize the clustering results
plt.figure(figsize=(14, 6))# K-means clustering result
plt.subplot(1, 2, 1)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans_labels, palette='viridis')
plt.title('K-means Clustering')# EM clustering result
plt.subplot(1, 2, 2)
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=gmm_labels, palette='viridis')
plt.title('EM Clustering (Gaussian Mixture Model)')plt.show()