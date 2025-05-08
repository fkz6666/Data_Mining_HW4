import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score
import numpy as np

# Set pandas display options for complete output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 20)

# 1. Load and prepare data
print("\n=== 1. Loading Data ===")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv(url, delim_whitespace=True, names=column_names, na_values='?')

print("\nFirst 5 rows of raw data:")
print(data.head())

# Select continuous feature fields
continuous_features = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']
X = data[continuous_features].copy()

# Impute missing values with mean
print("\nMissing value statistics (before imputation):")
print(X.isnull().sum())

imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("\nMissing value statistics (after imputation):")
print(X_imputed.isnull().sum())

# 2. Perform hierarchical clustering
print("\n=== 2. Performing Hierarchical Clustering ===")
agg_clustering = AgglomerativeClustering(
    n_clusters=3,
    affinity='euclidean',
    linkage='average',
    compute_full_tree=False
)

clusters = agg_clustering.fit_predict(X_imputed)
data['cluster'] = clusters

print("\nCluster assignment results (first 10 rows):")
print(data[['car_name', 'cluster', 'origin']].head(10))

# 3. Analyze clustering results
print("\n=== 3. Cluster Analysis ===")
# Calculate statistics for each cluster
cluster_stats = X_imputed.groupby(data['cluster']).agg(['mean', 'var', 'count'])

print("\nCluster statistics :")
print(cluster_stats)

# Calculate statistics by origin
origin_stats = X_imputed.groupby(data['origin']).agg(['mean', 'var', 'count'])

print("\nStatistics by origin :")
print(origin_stats)

# 4. Evaluate relationship between clusters and origin
print("\n=== 4. Cluster-Origin Relationship Evaluation ===")
# Create cross-tabulation
cross_tab = pd.crosstab(data['cluster'], data['origin'], normalize='index', margins=True)

print("\nCluster-origin correspondence (percentages):")
print(cross_tab)

# Calculate Adjusted Rand Index
ari = adjusted_rand_score(data['origin'], data['cluster'])
print(f"\nAdjusted Rand Index: {ari:.3f}")

# 5. Detailed analysis of each cluster
print("\n=== 5. Detailed Cluster Analysis ===")
for cluster in sorted(data['cluster'].unique()):
    cluster_data = data[data['cluster'] == cluster]
    print(f"\nDetailed statistics for Cluster {cluster} ({len(cluster_data)} cars):")

    # Origin distribution
    origin_dist = cluster_data['origin'].value_counts(normalize=True)
    print("\nOrigin distribution:")
    print(origin_dist)

    # Feature statistics
    print("\nFeature statistics:")
    print(cluster_data[continuous_features].describe())

    # Representative car examples
    print("\nRepresentative car examples:")
    print(cluster_data[['car_name', 'mpg', 'origin']].head(3))