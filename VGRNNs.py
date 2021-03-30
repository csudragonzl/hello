#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
# from sklearn.datasets import fetch_mldata
# from torch_geometric import nn as tgnn
from input_data import load_data
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges
import scipy.sparse as sp
from scipy.linalg import block_diag
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import tarfile
import torch.nn.functional as F
import copy
import time
import os
import networkx as nx
import scipy.io as sio

import inspect
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import copy
import pickle

from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.datasets import Planetoid

from clustering_metric import clustering_metrics
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from utils import *
from scipy.sparse import csr_matrix
from sklearn import metrics

from collections import defaultdict
import community

from sklearn.cluster import KMeans
# In[17]:
import torch_scatter
from torch_scatter import scatter_mean, scatter_max, scatter_add

# pip install torch_scatter==latest+cpu -f https://pytorch-geometric.com/whl/torch-1.4.0.html


class VGRNN_top:
    def __init__(self, filename):
        '''
        parser = argparse.ArgumentParser(
            description='Train VaDE with MNIST dataset',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--epochs', '-e',
                            help='Number of epochs.',
                            type=int, default=500)
        parser.add_argument('--T', type=int)
        parser.add_argument('--dataset-str', type=str, default='cora', help='type of dataset.')
        parser.add_argument('--featureflag', type=bool, default=False, help='featureflag')
        args = parser.parse_args()
        '''

        T_dict = {'HS11': 7, 'HS12': 8, 'workplace': 8, 'primary':6,'fix3': 10, 'fix5': 10, 'var3': 10, 'var5': 10, 'slashdot': 15, 'digg': 16, 'reality': 4, 'fbmessages': 6,
                  'mammalia-voles-bhp-trapping': 7, 'enron': 12, 'enronfeatures': 12, 'cellphone': 10}
        epoch_dict = {'HS11': 60, 'HS12': 80, 'workplace': 100, 'primary':100, 'fix3': 100, 'fix5': 5, 'var3': 10, 'var5': 10, 'slashdot': 300, 'digg': 300, 'reality': 1, 'fbmessages': 500,
                      'mammalia-voles-bhp-trapping': 600, 'enron': 500,'enronfeatures':300, 'cellphone': 500}
        N_dict = {'HS11': 126, 'HS12': 180, 'workplace': 92, 'primary':242, 'fix3': 128, 'fix5': 128, 'var3': 393, 'var5': 393, 'slashdot': 51083, 'digg': 30398, 'reality': 6809, 'fbmessages': 1899,
                  'mammalia-voles-bhp-trapping': 1686, 'enron': 151,'enronfeatures':151, 'cellphone': 400}
        self.T = T_dict[filename]  # args.dataset_str
        self.epoch = epoch_dict[filename]
        start = 1
        self.filename = filename  # 'fix3'

        '''
        cellphone
        enron
        fix3
        hs11
        hs12
        workplace
        primary
        LFRN500
        '''
        # creating edge list

        self.edge_idx_list = []
        self.adj_orig_dense_list, self.labels_l, self.N_CLASSES_l, x_in_list = [], [], [], []
        G_l = []
        for i in range(start, self.T + start):
            label_dict = {'HS11': 'E:/PycharmProjects/hello/data/HS11/snapshotall' + str(i) + '.txt',
                          'HS12': 'E:/PycharmProjects/hello/data/HS12/rere_comm_all' + str(i) + '.txt',
                          'primary': 'E:/PycharmProjects/hello/data/primary/comm_all' + str(i) + '.txt',
                          'workplace': 'E:/PycharmProjects/hello/data/workplace/rere_comm_all' + str(
                              i) + '.txt',
                          'var3': 'E:/PycharmProjects/hello/data/MinSoo_datasets/synvar/var3/synvar_3.t' + str(i) + '.comm1',
                          'var5': 'E:/PycharmProjects/hello/data/MinSoo_datasets/synvar/var5/synvar_5.t' + str(i) + '.comm1',
                          'fix5': 'E:/PycharmProjects/hello/data/MinSoo_datasets/synfix/fix5/synfix_5.t' + str(i) + '.comm1',
                          'fix3': 'E:/PycharmProjects/hello/data/MinSoo_datasets/synfix/fix3/synfix_3.t' + str(i) + '.comm1',
                          'LFRN500': 'F:/share/Dataset/LFR/N=500/re_switch.t' + str(i) + '.txt'}

            adj_dict = {'HS11': 'E:/PycharmProjects/hello/data/HS11/snapshot_edges_' + str(i) + '.txt',
                        'HS12': 'E:/PycharmProjects/hello/data/HS12/re_snapshot_' + str(i) + '_edges.txt',
                        'primary': 'E:/PycharmProjects/hello/data/primary/re_edges' + str(i) + '.txt',
                        'workplace': 'E:/PycharmProjects/hello/data/workplace/re_snapshot_edges_' + str(
                            i) + '.txt',
                        'var3': 'E:/PycharmProjects/hello/data/MinSoo_datasets/synvar/var3/synvar_3.t' + str(i) + '.edges',
                        'var5': 'E:/PycharmProjects/hello/data/MinSoo_datasets/synvar/var5/synvar_5.t' + str(i) + '.edges',
                        'fix5': 'E:/PycharmProjects/hello/data/MinSoo_datasets/synfix/fix5/synfix_5.t' + str(i) + '.edges',
                        'fix3': 'E:/PycharmProjects/hello/data/MinSoo_datasets/synfix/fix3/synfix_3.t' + str(i) + '.edges',
                        'LFRN500': 'F:/share/Dataset/LFR/N=500/switch.t' + str(i) + '.edges',
                        'slashdot': 'F:/share/Dataset/real world data/slashdot/edgelist' + str(i) + '.txt',
                        'digg': 'F:/share/Dataset/real world data/digg/edgelist' + str(i) + '.txt',
                        'reality': 'F:/share/Dataset/real world data/reality/edgelist' + str(i) + '.txt',
                        'cellphone': 'E:/PycharmProjects/hello/data/cellphone/cellphone' + str(i) + '.edges',
                        'enron':'E:/PycharmProjects/hello/data/enron/enron' + str(i) + '.edges',
                        'enronfeatures':'E:/PycharmProjects/hello/data/enron/enron.t' + str(i) + '.edges',
                        'fbmessages': 'E:/PycharmProjects/hello/data/fbmessages/edgelist' + str(i) + '.txt',
                        'mammalia-voles-bhp-trapping': 'E:/PycharmProjects/hello/data/voles/edgelist' + str(
                            i) + '.txt'}

            '''
                真相数据有labels
            labels = load_lab(label_dict[filename])#计算NMI时使用标签节点            
            labels = labels[:,1]
            labels = labels.astype(np.int16)
            self.labels_l.append(labels)
            l1 = list(set(labels))
            N_CLASSES = len(l1)
            self.N_CLASSES_l.append(N_CLASSES)
            N = len(labels)
            '''

            N = N_dict[filename]
            #adj_init, _, G = load_adj_edges(adj_dict[filename], N)
            adj_init, features, _, G = load_adj_edges(adj_dict[filename], N)
            '''
            x = np.identity(N, dtype=float)
            x = torch.FloatTensor(x)
            x_in_list.append(x)
            '''
            G_l.append(G)
            adj = csr_matrix(adj_init)
            fea = csr_matrix(features)#features
            edge = sparse_to_tuple(adj)[0]
            edge = np.array(edge)
            adj_label = adj
            adj_label = torch.FloatTensor(adj_label.toarray())

            fea_label = fea#features
            fea_label = torch.FloatTensor(fea_label.toarray())#features
            print(fea_label)
            #x_in_list.append(adj_label)
            x_in_list.append(fea_label)

            self.adj_orig_dense_list.append(adj_label)
            self.edge_idx_list.append(torch.tensor(np.transpose(edge), dtype=torch.long))



        self.x_in = Variable(torch.stack(x_in_list))

        for i in range(self.T):
            part = community.best_partition(G_l[i])
            key_max = max(part.keys(), key=(lambda k: part[k]))
            self.N_CLASSES_l.append(part[key_max] + 1)


    def VGRNN_main(self):
        h_dim = 32
        z_dim = 16
        n_layers = 1
        clip = 10
        learning_rate = 1e-2
        seq_len = self.T

        num_nodes = self.adj_orig_dense_list[seq_len - 1].shape[0]

        x_dim = num_nodes

        eps = 1e-10
        conv_type = 'GCN'

        '''
        adj_label_l = []
        for i in range(len(adj_train_l)):
            temp_matrix = adj_train_l[i]

            adj_label_l.append(torch.tensor(temp_matrix.toarray().astype(np.float32)))
        '''
        # building model

        model = VGRNN(x_dim, h_dim, z_dim, n_layers, eps, conv=conv_type, bias=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # training

        seq_start = 0
        seq_end = self.T
        tst_after = 0

        for k in range(self.epoch):
            optimizer.zero_grad()
            start_time = time.time()
            com_loss = 0
            kld_loss, nll_loss,com_loss, _, _, hidden_st, h_l, all_z_t = model(self.x_in[seq_start:seq_end]
                                                                      , self.edge_idx_list[seq_start:seq_end]
                                                                      , self.adj_orig_dense_list[seq_start:seq_end])

            loss = kld_loss + nll_loss + com_loss
            #loss = nll_loss#back to GAE

            loss.backward()
            optimizer.step()

            # nn.utils.clip_grad_norm(model.parameters(), clip)

            print('epoch: ', k)

            print('loss =', loss.mean().item())
            print('----------------------------------')

            with torch.no_grad():
                _, _, _, _, _, hidden_st, h_l, all_z_t = model(self.x_in[seq_start:seq_end]
                                                            , self.edge_idx_list[seq_start:seq_end]
                                                            , self.adj_orig_dense_list[seq_start:seq_end])
            # , hidden_st)

            for i in range(self.T):
                gmm = GaussianMixture(n_components=self.N_CLASSES_l[i], covariance_type='diag')
                # pre1 = KMeans(n_clusters = N_CLASSES_l[i], init = 'k-means++').fit(all_z_t[i]).labels_

                pre = gmm.fit_predict(all_z_t[i])  # (h_l[i][0])


                Q_res = Modularity(self.adj_orig_dense_list[i], pre)
                print("Q: " + str(Q_res))
                '''
                #有真相数据计算NMI，原先注释这三行
            nmi = metrics.normalized_mutual_info_score(self.labels_l[i], pre)
            print("nmi: "+str(nmi))
        print(model.state_dict())                
                '''
        for i in range(self.T):
            Q_res, A_res, N_res, I_res = [], [], [], []
            Nmax = -222
            for j in range(20):
                gmm = GaussianMixture(n_components=self.N_CLASSES_l[i], covariance_type='diag')
                pre = gmm.fit_predict(all_z_t[i])
                '''
                计算NMI
                cm = clustering_metrics(self.labels_l[i], pre)
                A_res,N_res,I_res=cm.evaluationClusterModelFromLabel(tqdm,self.filename,A_res,N_res,I_res)                
                '''

                tem = Modularity(self.adj_orig_dense_list[i], pre)
                Q_res.append(tem)
                print("Q: " + str(tem))
                '''
                NMI
                if Nmax < N_res[j]:
                    Nmax = N_res[j]
                    pre_max = pre
                '''
                '''
                模块度
                '''
                if Nmax < Q_res[j]:
                    Nmax = Q_res[j]
                    pre_max = pre

            if i == 0:
                clu_res = pre_max
            else:
                clu_res = np.c_[clu_res, pre_max]

            root_dir = os.path.abspath(os.path.dirname(__file__))
            path = root_dir + '\Result\\'
            '''
            NMI部分值
            fh = open(path+self.filename+'_evalu.txt', 'a')
            fh.write('%f	%f' % (max(A_res),max(N_res)))
            fh.write('\n')
            fh.flush()
            fh.close()
            '''

            '''
            模块度
            '''
            fh = open(path+self.filename + '_evalu1.txt', 'a')
            fh.write('%f' % (max(Q_res)))
            fh.write('\n')
            fh.flush()
            fh.close()

        np.savetxt(path + self.filename + "_clu1.txt", clu_res, fmt="%d", )



# utility functions
def uniform(size, tensor):
    stdv = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def glorot(tensor):
    stdv = math.sqrt(6.0 / (tensor.size(0) + tensor.size(1)))
    if tensor is not None:
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def scatter_(name, src, index, dim_size=None):
    r"""Aggregates all values from the :attr:`src` tensor at the indices
    specified in the :attr:`index` tensor along the first dimension.
    If multiple indices reference the same location, their contributions
    are aggregated according to :attr:`name` (either :obj:`"add"`,
    :obj:`"mean"` or :obj:`"max"`).
    Args:
        name (string): The aggregation to use (:obj:`"add"`, :obj:`"mean"`,
            :obj:`"max"`).
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements to scatter.
        dim_size (int, optional): Automatically create output tensor with size
            :attr:`dim_size` in the first dimension. If set to :attr:`None`, a
            minimal sized output tensor is returned. (default: :obj:`None`)
    :rtype: :class:`Tensor`
    """
    assert name in ['add', 'mean', 'max']

    op = getattr(torch_scatter, 'scatter_{}'.format(name))
    fill_value = -1e38 if name is 'max' else 0

    out = op(src, index, 0, None, dim_size)  # fill_value
    if isinstance(out, tuple):
        out = out[0]

    if name is 'max':
        out[out == fill_value] = 0

    return out


class MessagePassing(torch.nn.Module):
    r"""Base class for creating message passing layers
    .. math::
        \mathbf{x}_i^{\prime} = \gamma_{\mathbf{\Theta}} \left( \mathbf{x}_i,
        \square_{j \in \mathcal{N}(i)} \, \phi_{\mathbf{\Theta}}
        \left(\mathbf{x}_i, \mathbf{x}_j,\mathbf{e}_{i,j}\right) \right),
    where :math:`\square` denotes a differentiable, permutation invariant
    function, *e.g.*, sum, mean or max, and :math:`\gamma_{\mathbf{\Theta}}`
    and :math:`\phi_{\mathbf{\Theta}}` denote differentiable functions such as
    MLPs.
    See `here <https://rusty1s.github.io/pytorch_geometric/build/html/notes/
    create_gnn.html>`__ for the accompanying tutorial.
    """

    def __init__(self, aggr='add'):
        super(MessagePassing, self).__init__()

        self.message_args = inspect.getfullargspec(self.message)[0][1:]
        self.update_args = inspect.getfullargspec(self.update)[0][2:]

    def propagate(self, aggr, edge_index, **kwargs):
        r"""The initial call to start propagating messages.
        Takes in an aggregation scheme (:obj:`"add"`, :obj:`"mean"` or
        :obj:`"max"`), the edge indices, and all additional data which is
        needed to construct messages and to update node embeddings."""

        assert aggr in ['add', 'mean', 'max']
        kwargs['edge_index'] = edge_index

        size = None
        message_args = []
        for arg in self.message_args:
            if arg[-2:] == '_i':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[0]])
            elif arg[-2:] == '_j':
                tmp = kwargs[arg[:-2]]
                size = tmp.size(0)
                message_args.append(tmp[edge_index[1]])
            else:
                message_args.append(kwargs[arg])

        update_args = [kwargs[arg] for arg in self.update_args]

        out = self.message(*message_args)
        out = scatter_(aggr, out, edge_index[0], dim_size=size)
        out = self.update(out, *update_args)

        return out

    def message(self, x_j):  # pragma: no cover
        r"""Constructs messages in analogy to :math:`\phi_{\mathbf{\Theta}}`
        for each edge in :math:`(i,j) \in \mathcal{E}`.
        Can take any argument which was initially passed to :meth:`propagate`.
        In addition, features can be lifted to the source node :math:`i` and
        target node :math:`j` by appending :obj:`_i` or :obj:`_j` to the
        variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`."""

        return x_j

    def update(self, aggr_out):  # pragma: no cover
        r"""Updates node embeddings in analogy to
        :math:`\gamma_{\mathbf{\Theta}}` for each node
        :math:`i \in \mathcal{V}`.
        Takes in the output of aggregation as first argument and any argument
        which was initially passed to :meth:`propagate`."""

        return aggr_out


def tuple_to_array(lot):
    out = np.array(list(lot[0]))
    for i in range(1, len(lot)):
        out = np.vstack((out, np.array(list(lot[i]))))

    return out


# In[23]:
# layers
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, act=F.relu, improved=True, bias=False):
        super(GCNConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.act = act

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, aaa = add_self_loops(edge_index, num_nodes=x.size(0))

        loop_weight = torch.full(
            (x.size(0),),
            1 if not self.improved else 2,
            dtype=x.dtype,
            device=x.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

        row, col = edge_index

        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv = deg.pow(-0.5)
        deg_inv[deg_inv == float('inf')] = 0

        norm = deg_inv[row] * edge_weight * deg_inv[col]

        x = torch.matmul(x, self.weight)
        out = self.propagate('add', edge_index, x=x, norm=norm)
        return self.act(out)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class graph_gru_gcn(nn.Module):
    def __init__(self, input_size, hidden_size, n_layer, bias=True):
        super(graph_gru_gcn, self).__init__()

        self.hidden_size = hidden_size
        self.n_layer = n_layer

        # gru weights
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []

        for i in range(self.n_layer):
            if i == 0:
                self.weight_xz.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(input_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
            else:
                self.weight_xz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hz.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hr.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_xh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))
                self.weight_hh.append(GCNConv(hidden_size, hidden_size, act=lambda x: x, bias=bias))

    def forward(self, inp, edgidx, h):
        h_out = torch.zeros(h.size())
        for i in range(self.n_layer):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](inp, edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](inp, edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](inp, edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
            #		  out = self.decoder(h_t.view(1,-1))
            else:
                z_g = torch.sigmoid(self.weight_xz[i](h_out[i - 1], edgidx) + self.weight_hz[i](h[i], edgidx))
                r_g = torch.sigmoid(self.weight_xr[i](h_out[i - 1], edgidx) + self.weight_hr[i](h[i], edgidx))
                h_tilde_g = torch.tanh(self.weight_xh[i](h_out[i - 1], edgidx) + self.weight_hh[i](r_g * h[i], edgidx))
                h_out[i] = z_g * h[i] + (1 - z_g) * h_tilde_g
        #		  out = self.decoder(h_t.view(1,-1))

        out = h_out
        return out, h_out


class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()

        self.act = act
        self.dropout = dropout

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)


# VGRNN model

class VGRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, n_layers, eps, conv, bias=False):
        super(VGRNN, self).__init__()

        self.x_dim = x_dim
        self.eps = eps
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        if conv == 'GCN':
            self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
            self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())

            # self.enc = GCNConv(h_dim, h_dim)
            self.enc = GCNConv(h_dim + h_dim, h_dim)
            self.enc_mean = GCNConv(h_dim, z_dim, act=lambda x: x)
            self.enc_std = GCNConv(h_dim, z_dim, act=F.softplus)

            self.prior = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ReLU())
            self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
            self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

            self.rnn = graph_gru_gcn(h_dim + h_dim, h_dim, n_layers, bias)

    def forward(self, x, edge_idx_list, adj_orig_dense_list, hidden_in=None):
        assert len(adj_orig_dense_list) == len(edge_idx_list)

        kld_loss = 0
        nll_loss = 0
        com_loss = 0
        all_enc_mean, all_enc_std = [], []
        all_prior_mean, all_prior_std = [], []
        all_dec_t, all_z_t = [], []
        h_l = []
        if hidden_in is None:
            h = Variable(torch.zeros(self.n_layers, x.size(1), self.h_dim))
        else:
            h = Variable(hidden_in)

        for t in range(x.size(0)):
            phi_x_t = self.phi_x(x[t])

            # encoder
            enc_t = self.enc(torch.cat([phi_x_t, h[-1]], 1), edge_idx_list[t])#退化成GAE后注释掉
            #enc_t = self.enc(phi_x_t, edge_idx_list[t])
            enc_mean_t = self.enc_mean(enc_t, edge_idx_list[t])
            enc_std_t = self.enc_std(enc_t, edge_idx_list[t])

            # prior
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)

            '''
            退化成GAE
            z_t = enc_mean_t
            '''
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_t = self.dec(z_t)

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1), edge_idx_list[t], h)
            h_l.append(h)

            nnodes = adj_orig_dense_list[t].size()[0]
            enc_mean_t_sl = enc_mean_t[0:nnodes, :]
            enc_std_t_sl = enc_std_t[0:nnodes, :]
            prior_mean_t_sl = prior_mean_t[0:nnodes, :]
            prior_std_t_sl = prior_std_t[0:nnodes, :]
            dec_t_sl = dec_t[0:nnodes, 0:nnodes]

            # computing losses
            kld_loss += self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, prior_mean_t_sl, prior_std_t_sl)
            nll_loss += self._nll_bernoulli(dec_t_sl, adj_orig_dense_list[t])
            if t > 0:
                com_loss += 0.3 * self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, enc_mean_t_1_sl, enc_std_t_1_sl)

            all_enc_std.append(enc_std_t_sl)#退化成GAE后注释掉
            all_enc_mean.append(enc_mean_t_sl)
            all_prior_mean.append(prior_mean_t_sl)
            all_prior_std.append(prior_std_t_sl)#退化成GAE后注释掉
            all_dec_t.append(dec_t_sl)
            all_z_t.append(z_t)
            enc_mean_t_1_sl = enc_mean_t_sl
            enc_std_t_1_sl = enc_std_t_sl

        return kld_loss, nll_loss,com_loss, all_enc_mean, all_prior_mean, h, h_l, all_z_t

    def dec(self, z):
        outputs = InnerProductDecoder(act=lambda x: x)(z)
        return outputs

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def _reparameterized_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_()
        eps1 = Variable(eps1)
        return eps1.mul(std).add_(mean)

    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element = (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                       (torch.pow(std_1 + self.eps, 2) + torch.pow(mean_1 - mean_2, 2)) /
                       torch.pow(std_2 + self.eps, 2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)

    def _kld_gauss_zu(self, mean_in, std_in):
        num_nodes = mean_in.size()[0]
        std_log = torch.log(std_in + self.eps)
        kld_element = torch.mean(torch.sum(1 + 2 * std_log - mean_in.pow(2) -
                                           torch.pow(torch.exp(std_log), 2), 1))
        return (-0.5 / num_nodes) * kld_element

    def _nll_bernoulli(self, logits, target_adj_dense):
        temp_size = target_adj_dense.size()[0]
        temp_sum = target_adj_dense.sum()
        posw = float(temp_size * temp_size - temp_sum) / temp_sum
        norm = temp_size * temp_size / float((temp_size * temp_size - temp_sum) * 2)
        nll_loss_mat = F.binary_cross_entropy_with_logits(input=logits
                                                          , target=target_adj_dense
                                                          , pos_weight=posw
                                                          , reduction='none')
        nll_loss = -1 * norm * torch.mean(nll_loss_mat, dim=[0, 1])
        return - nll_loss


# hyperparameters


#model = VGRNN_top('mammalia-voles-bhp-trapping')  # voles
#model = VGRNN_top('fbmessages')  # messages
#model = VGRNN_top('enron')  # enron
model = VGRNN_top('enronfeatures')
#model = VGRNN_top('cellphone')  # cellphone
#model = VGRNN_top('HS11')  # HS11
#model = VGRNN_top('HS12')  # HS12
#model = VGRNN_top('workplace')  # workplace
#model = VGRNN_top('primary')  # primary
#model = VGRNN_top('fix3')  # fix3
#model = VGRNN_top('fix5')  # fix5
#model = VGRNN_top('var3')  # var3
#model = VGRNN_top('var5')  # var5
model.VGRNN_main()
