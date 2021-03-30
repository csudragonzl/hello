import numpy as np
import pandas as pd
import networkx as nx

def load_adj_edges(directory, N):
    id_to_node = {}
    node_to_id = {}
    weight = {}#权重矩阵

    # Load csv file
    edges = pd.read_csv(directory, delim_whitespace=True, header=None)
    edges = edges.values
    #edges = [tuple(edge) for edge in edges]
    edges = np.array(edges)
    tmp = edges[:, 0:2]
    G = nx.Graph()
    G.add_edges_from(tmp)

    # Build adjacency matrix
    adjacency = np.zeros((N, N), dtype=int)
    features = np.zeros((N, N), dtype=int)
    # adjacency = np.zeros((len(G.nodes()), len(G.nodes())), dtype=int)
    for edge in edges:
        adjacency[edge[0] - 1, edge[1] - 1] = 1
        adjacency[edge[1] - 1, edge[0] - 1] = 1#when the graph is undirected
        features[edge[0] - 1, edge[1] - 1] = edge[2]
        features[edge[1] - 1, edge[0] - 1] = edge[2]

    # adjacency = adjacency + np.identity(len(G.nodes()))
    return adjacency, features, edges, G

if __name__ == "__main__":
    adjacency, features, edges, G = load_adj_edges('E:\PycharmProjects\hello\data\enron\enron.t01.edges', 151)
    print(adjacency)
    print(features)