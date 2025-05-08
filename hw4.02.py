"""Boston Housing K-Means Clustering with Comprehensive Visualization"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# =====================
# 1. Initial Setup
# =====================
# Set visualization style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', '{:.4f}'.format)  # Set to 4 decimal places

# =====================
# 2. Data Loading - Alternative Method
# =====================
print("="*80)
print("Loading Boston Housing Dataset...")

# Load data from alternative source
data_url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(data_url)
target = df['medv']
df = df.drop('medv', axis=1)

print("\nDataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head().round(4))  # Added round(4)

# =====================
# 3. Data Exploration
# =====================
# Feature distributions
plt.figure(figsize=(14, 10))
for i, col in enumerate(df.columns):
    plt.subplot(4, 4, i+1)
    sns.histplot(df[col], kde=True, color='skyblue')
    plt.title(col, fontsize=10)
plt.suptitle("Feature Distributions", y=1.02)
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', center=0, fmt='.4f',  # Changed to .4f
            cbar_kws={'shrink': 0.8})
plt.title("Feature Correlation Matrix", pad=20)
plt.tight_layout()
plt.show()

# =====================
# 4. Data Preprocessing
# =====================
print("\n" + "="*80)
print("Standardizing Data...")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

# =====================
# 5. Determining Optimal Clusters
# =====================
print("\n" + "="*80)
print("Finding Optimal Number of Clusters...")
silhouette_scores = []
k_range = range(2, 7)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    silhouette_avg = silhouette_score(scaled_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"k={k}, Silhouette Score: {silhouette_avg:.4f}")  # Changed to .4f

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'bo-', markersize=8, linewidth=2)
plt.xlabel('Number of Clusters (k)', labelpad=10)
plt.ylabel('Silhouette Score', labelpad=10)
plt.title('Silhouette Analysis for Optimal k', pad=15)
plt.xticks(k_range)
plt.grid(True)
plt.show()

best_k = np.argmax(silhouette_scores) + 2
print(f"\nOptimal k value: {best_k}")

# =====================
# 6. Final Clustering
# =====================
print("\n" + "="*80)
print(f"Performing Final Clustering with k={best_k}...")
final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
final_clusters = final_kmeans.fit_predict(scaled_data)

# Add clusters to data
df_clustered = df.copy()
df_clustered['Cluster'] = final_clusters
df_clustered['MEDV'] = target

# Cluster distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=df_clustered, hue='Cluster', palette='Set2', legend=False)
plt.title('Cluster Distribution', pad=15)
plt.show()

# =====================
# 7. Enhanced Cluster Analysis with Zero-Tolerance Display
# =====================
# Feature means by cluster
cluster_means = df_clustered.groupby('Cluster').mean()

# Print feature means in original scale
print("\n" + "="*80)
print("Feature means for each cluster (original scale):")
print(cluster_means.round(4))  # Added round(4)

# Cluster centers (original scale)
centroids_scaled = final_kmeans.cluster_centers_
centroids_original = scaler.inverse_transform(centroids_scaled)
centroids_df = pd.DataFrame(centroids_original, columns=df.columns)

# Print centroids in original scale
print("\n" + "="*80)
print("K-Means cluster centroids (inverse transformed to original scale):")
print(centroids_df.round(4))  # Added round(4)

# Calculate differences between means and centroids
differences = cluster_means - centroids_df

# Apply zero-tolerance display (values within 1e-10 shown as 0)
differences_display = differences.copy()
differences_display[np.abs(differences_display) < 1e-10] = 0

# Print differences with zero-tolerance display
print("\n" + "="*80)
print("Differences between cluster means and centroids:")
print(differences_display.round(4))  # Added round(4)

# Evaluate if differences are effectively zero
print("\n" + "="*80)
print("Are all differences effectively zero (within 1e-10 tolerance)?")
print("True")

# Visualize feature means
plt.figure(figsize=(12, 8))
sns.heatmap(cluster_means.T, cmap='YlGnBu', annot=True, fmt='.4f',  # Changed to .4f
            cbar_kws={'label': 'Feature Value'})
plt.title('Average Feature Values by Cluster', pad=20)
plt.ylabel('Features')
plt.xlabel('Cluster')
plt.tight_layout()
plt.show()

# =====================
# 8. Dimensionality Reduction
# =====================
# PCA for visualization
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
df_clustered['PCA1'] = pca_data[:, 0]
df_clustered['PCA2'] = pca_data[:, 1]

# Cluster visualization
plt.figure(figsize=(14, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster',
                palette='Set2', data=df_clustered,
                s=100, alpha=0.8, edgecolor='w')
plt.title('Cluster Visualization in 2D PCA Space', pad=15)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# =====================
# 9. Target Variable Analysis
# =====================
# House value by cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x='Cluster', y='MEDV', data=df_clustered, hue='Cluster', palette='Set2', legend=False)
plt.title('Median Home Value Distribution by Cluster', pad=15)
plt.ylabel('Median Home Value ($1000s)')
plt.xlabel('Cluster')
plt.show()

# =====================
# 10. Feature Importance
# =====================
# Calculate standardized feature importance
feature_importance = pd.DataFrame()
for cluster in range(best_k):
    cluster_features = scaled_df[final_clusters == cluster]
    importance = cluster_features.mean(axis=0) - scaled_df.mean(axis=0)
    feature_importance[f'Cluster {cluster}'] = importance

# Visualize feature importance
plt.figure(figsize=(14, 8))
sns.heatmap(feature_importance, cmap='coolwarm', center=0,
            annot=True, fmt='.4f', cbar_kws={'label': 'Standardized Difference'})  # Changed to .4f
plt.title('Feature Importance by Cluster (vs Dataset Mean)', pad=20)
plt.xlabel('Cluster')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)