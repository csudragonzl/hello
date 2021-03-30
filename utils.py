import pickle as pkl
import pandas as pd
import os
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import matplotlib.pyplot as plt

def load_lab_mongo(loader, i):
    content = loader.communities[i]
    return content


def load_lab(directory):
    content = np.loadtxt(directory)
    return content


def load_adj_edges_mongo(loader, i, N):
    G = loader.snapshots[i]
    adjacency = np.zeros((N, N), dtype=int)
    edges = G.get_edgelist()
    for edge in edges:
        adjacency[edge[0], edge[1]] = 1
        adjacency[edge[1], edge[0]] = 1
    return adjacency, edges, G

'''
#no features in data
def load_adj_edges(directory, N):
    id_to_node = {}
    node_to_id = {}

    # Load csv file
    edges = pd.read_csv(directory, delim_whitespace=True, header=None)
    edges = edges.values
    edges = [tuple(edge) for edge in edges]

    G = nx.Graph()
    G.add_edges_from(edges)
    # Build adjacency matrix

    adjacency = np.zeros((N, N), dtype=int)
    # adjacency = np.zeros((len(G.nodes()), len(G.nodes())), dtype=int)
    for edge in edges:
        adjacency[edge[0] - 1, edge[1] - 1] = 1
        adjacency[edge[1] - 1, edge[0] - 1] = 1#when the graph is undirected

    # adjacency = adjacency + np.identity(len(G.nodes()))
    return adjacency, edges, G
'''
#with features in data
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

# python2.7
# type(graph) = networkx.Graph
# D = nx.DiGraph(a)
def Modularity(adj, result):
    adj = adj.numpy()

    l1 = list(set(result))
    NCLUST = len(l1)
    result = np.array(result)
    TS = np.sum(adj) / 2.0;
    Q = 0;

    for i in range(NCLUST):
        tem = np.where(result == i)
        IS = adj[tem]
        IS = IS[:, tem]
        IS = np.sum(IS) / 2.0
        DS = np.sum(adj[tem, :])
        Q = Q + IS / TS - (DS / (2 * TS)) ** 2;

    return Q


def load_data(dataset):
    # load the data: x, tx, allx, graph
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        '''
        fix Pickle incompatibility of numpy arrays between Python 2 and 3
        https://stackoverflow.com/questions/11305790/pickle-incompatibility-of-numpy-arrays-between-python-2-and-3
        '''
        with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as rf:
            u = pkl._Unpickler(rf)
            u.encoding = 'latin1'
            cur_data = u.load()
            objects.append(cur_data)
    # objects.append(
    #	pkl.load(open("data/ind.{}.{}".format(dataset, names[i]), 'rb')))
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = torch.FloatTensor(np.array(features.todense()))
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features


def load_label(dataset):
    """ Load node-level labels from tkipf/gae input files
    :param dataset: 'cora', 'citeseer' or 'pubmed' graph dataset.
    :return: n-dim array of node labels (used for clustering)
    """
    names = ['ty', 'ally']
    objects = []
    for i in range(len(names)):
        with open("./data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))
    ty, ally = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    label = sp.vstack((ally, ty)).tolil()
    label[test_idx_reorder, :] = label[test_idx_range, :]
    # One-hot to integers
    label = np.argmax(label.toarray(), axis=1)
    return label


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


def mask_test_edges(adj):
    # Function to build test set with 10% positive links
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    # TODO: Clean up.

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]
    num_test = int(np.floor(edges.shape[0] / 10.))
    num_val = int(np.floor(edges.shape[0] / 20.))

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], train_edges):
            continue
        if ismember([idx_j, idx_i], train_edges):
            continue
        if ismember([idx_i, idx_j], val_edges):
            continue
        if ismember([idx_j, idx_i], val_edges):
            continue
        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])

    assert ~ismember(test_edges_false, edges_all)
    assert ~ismember(val_edges_false, edges_all)
    assert ~ismember(val_edges, train_edges)
    assert ~ismember(test_edges, train_edges)
    assert ~ismember(val_edges, test_edges)

    data = np.ones(train_edges.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def get_roc_score(emb, adj_orig, edges_pos, edges_neg):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


