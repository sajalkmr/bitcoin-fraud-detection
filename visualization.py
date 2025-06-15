import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.preprocessing import StandardScaler

def create_correlation_heatmap():
    # Read the features file
    print("Reading features file...")
    features_df = pd.read_csv('dataset/elliptic_txs_features.csv')
    
    # Get feature columns (excluding the first two columns which are txId and time_step)
    feature_cols = features_df.columns[2:]
    
    # Calculate correlation matrix
    print("Calculating correlation matrix...")
    corr_matrix = features_df[feature_cols].corr()
    
    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Set up the matplotlib figure
    plt.figure(figsize=(20, 16))
    
    # Create heatmap
    print("Creating heatmap...")
    sns.heatmap(corr_matrix, 
                mask=mask,
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=.5,
                cbar_kws={"shrink": .5})
    
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation heatmap saved to figures/correlation_heatmap.png")

def create_network_visualization():
    # Read the data
    print("Reading data files...")
    edges_df = pd.read_csv('dataset/elliptic_txs_edgelist.csv')
    classes_df = pd.read_csv('dataset/elliptic_txs_classes.csv')
    
    # Create a sample of the data for visualization (to avoid memory issues)
    sample_size = 1000
    edges_sample = edges_df.sample(n=sample_size, random_state=42)
    
    # Create a graph
    print("Creating network graph...")
    G = nx.Graph()
    
    # Add edges
    for _, row in edges_sample.iterrows():
        G.add_edge(row['txId1'], row['txId2'])
    
    # Get node colors based on class
    node_colors = []
    for node in G.nodes():
        class_info = classes_df[classes_df['txId'] == node]['class'].values
        if len(class_info) > 0:
            if class_info[0] == '1':  # fraudulent
                node_colors.append('red')
            elif class_info[0] == '2':  # legitimate
                node_colors.append('green')
            else:  # unknown
                node_colors.append('gray')
        else:
            node_colors.append('gray')
    
    # Create the plot
    plt.figure(figsize=(15, 15))
    
    # Use spring layout for node positions
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw the network
    nx.draw(G, pos,
            node_color=node_colors,
            node_size=50,
            alpha=0.6,
            with_labels=False)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Fraudulent',
               markerfacecolor='red', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Legitimate',
               markerfacecolor='green', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Unknown',
               markerfacecolor='gray', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title('Bitcoin Transaction Network', fontsize=16)
    plt.savefig('figures/transaction_network.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Network visualization saved to figures/transaction_network.png")

if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    import os
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Generate visualizations
    create_correlation_heatmap()
    create_network_visualization() 