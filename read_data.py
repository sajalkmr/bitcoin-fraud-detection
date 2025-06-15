import pandas as pd

# Read the first few rows of each file
print("\nClasses file:")
classes_df = pd.read_csv('dataset/elliptic_txs_classes.csv', nrows=5)
print(classes_df.head())

print("\nFeatures file:")
features_df = pd.read_csv('dataset/elliptic_txs_features.csv', nrows=5)
print(features_df.head())

print("\nEdgelist file:")
edges_df = pd.read_csv('dataset/elliptic_txs_edgelist.csv', nrows=5)
print(edges_df.head())

# Print basic information about each dataset
print("\nClasses file info:")
print(classes_df.info())

print("\nFeatures file info:")
print(features_df.info())

print("\nEdgelist file info:")
print(edges_df.info()) 