"""Wine Dataset K-Means Analysis with Comprehensive Visualization"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    homogeneity_score,
    completeness_score,
    silhouette_score,
    adjusted_rand_score
)
from sklearn.decomposition import PCA

# =====================
# 1. Setup and Configuration
# =====================
# Set style for plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# =====================
# 2. Data Loading and Initial Exploration
# =====================
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['true_label'] = wine.target

# Display basic dataset info
print("=" * 80)
print("Dataset Dimensions:", df.shape)
print("Features:", wine.feature_names)
print("Class Distribution:\n", df['true_label'].value_counts().sort_index())
print("=" * 80)

# Feature distribution visualization
plt.figure(figsize=(14, 10))
for i, feature in enumerate(wine.feature_names[:6]):
    plt.subplot(2, 3, i + 1)
    sns.boxplot(x='true_label', y=feature, data=df, palette="Set2")
    plt.title(feature)
plt.suptitle("Feature Distributions by True Wine Class", y=1.02)
plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
corr_matrix = df.iloc[:, :-1].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title("Feature Correlation Matrix")
plt.show()

# =====================
# 3. Data Preprocessing
# =====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop('true_label', axis=1))

# =====================
# 4. K-Means Clustering
# =====================
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Cluster centers visualization
centers_scaled = pd.DataFrame(
    kmeans.cluster_centers_,
    columns=df.columns[:-2]  # Exclude true_label and cluster
)

plt.figure(figsize=(12, 6))
sns.heatmap(centers_scaled, cmap="YlGnBu", annot=True, fmt=".2f")
plt.title("Cluster Centers in Standardized Feature Space")
plt.show()

# =====================
# 5. Evaluation Metrics
# =====================
metrics = {
    "Homogeneity": homogeneity_score(df['true_label'], df['cluster']),
    "Completeness": completeness_score(df['true_label'], df['cluster']),
    "Silhouette Score": silhouette_score(X_scaled, df['cluster']),
    "Adjusted Rand Index": adjusted_rand_score(df['true_label'], df['cluster'])
}

print("\nClustering Performance Evaluation:")
for name, value in metrics.items():
    print(f"{name}: {value:.4f}")
print("=" * 80)

# Metrics visualization
plt.figure(figsize=(10, 6))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
plt.bar(metrics.keys(), metrics.values(), color=colors)
plt.title("Clustering Performance Metrics", pad=20)
plt.ylim(0, 1)
for i, v in enumerate(metrics.values()):
    plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12)
plt.show()

# =====================
# 6. Dimensionality Reduction and Visualization
# =====================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]

# True vs Predicted visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# True labels plot
sns.scatterplot(
    x='pca1', y='pca2',
    hue='true_label',
    palette='viridis',
    data=df,
    s=100,
    ax=ax1
)
ax1.set_title("True Wine Classes (PCA Projection)", pad=15)
ax1.legend(title="True Class")

# Cluster assignments plot
sns.scatterplot(
    x='pca1', y='pca2',
    hue='cluster',
    palette='Set2',
    data=df,
    s=100,
    ax=ax2
)
ax2.set_title("K-Means Clusters (k=3)", pad=15)
ax2.legend(title="Cluster")

plt.suptitle("Comparison of True Classes and Cluster Assignments", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# =====================
# 7. Cluster Characteristics Analysis
# =====================
# Feature importance by cluster
cluster_features = df.groupby('cluster').mean().drop(['true_label', 'pca1', 'pca2'], axis=1)

plt.figure(figsize=(14, 6))
sns.heatmap(cluster_features.T, cmap="YlGnBu", annot=True, fmt=".1f")
plt.title("Average Feature Values by Cluster", pad=20)
plt.show()

# =====================
# 8. Optimal K Determination
# =====================
inertia = []
silhouette = []
k_range = range(2, 8)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_scaled, kmeans.labels_))

# Elbow and silhouette plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Elbow plot
ax1.plot(k_range, inertia, 'bo-', markersize=8, linewidth=2)
ax1.set_xlabel('Number of clusters (k)', labelpad=10)
ax1.set_ylabel('Inertia', labelpad=10)
ax1.set_title('Elbow Method for Optimal k', pad=15)
ax1.set_xticks(k_range)

# Silhouette plot
ax2.plot(k_range, silhouette, 'go-', markersize=8, linewidth=2)
ax2.set_xlabel('Number of clusters (k)', labelpad=10)
ax2.set_ylabel('Silhouette Score', labelpad=10)
ax2.set_title('Silhouette Analysis for Optimal k', pad=15)
ax2.set_xticks(k_range)

plt.suptitle("Determining the Optimal Number of Clusters", y=1.02, fontsize=14)
plt.tight_layout()
plt.show()

# =====================
# 9. Metric Explanations
# =====================
print("\n" + "=" * 80)
print("Metric Explanations:".center(80))
print("=" * 80)