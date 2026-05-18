from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import normalize,sparse_mx_to_torch_sparse_tensor
import scipy.sparse as sp

from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix


def update_graph(model, optimizer, features, adj, rew_states, loss, args, envs):
    if adj.shape[0] > 1:
        labels = torch.zeros((len(features)))
        idx_train = torch.LongTensor([0])
        for r_s in rew_states:
            if len(envs.observation_space.shape) == 1 : #MuJoCo experiments
                labels[r_s[0]] = torch.sigmoid(2*r_s[1])
            else:
                labels[r_s[0]] = torch.tensor([1.]) if r_s[1] > 0. else torch.tensor([0.])
            idx_train=torch.cat((idx_train, torch.LongTensor([r_s[0]]) ), 0)
        labels= labels.type(torch.LongTensor)
    else:
        labels = torch.zeros((len(features))).type(torch.LongTensor)
        idx_train = torch.LongTensor([0])
    
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.diag(adj.toarray().sum(axis=1))
    laplacian = torch.from_numpy((deg - adj.toarray()).astype(np.float32))
    adj = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    if args.cuda and torch.cuda.is_available():
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        laplacian =laplacian.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()

    t_total = time.time()
    for epoch in range(args.gcn_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        # assert not torch.isnan(output).any(), "Model output contains NaN!"
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        soft_out = torch.unsqueeze(torch.nn.functional.softmax(output,dim=1)[:,1], 1)
        loss_reg = torch.mm(torch.mm(soft_out.T,laplacian), soft_out)
        loss_train += args.gcn_lambda * loss_reg.squeeze()
        loss_train.backward()
        optimizer.step()


def update_graph_sat(model, optimizer, features, adj, rew_states, loss, args, envs, node_ptrs):
    if adj.shape[0] > 1:
        labels = torch.zeros((len(features)))
        idx_train = torch.LongTensor([0])
        for r_s in rew_states:
            if len(envs.observation_space.shape) == 1 : #MuJoCo experiments
                labels[r_s[0]] = torch.sigmoid(2*r_s[1])
            else:
                labels[r_s[0]] = torch.tensor([1.]) if r_s[1] > 0. else torch.tensor([0.])
            idx_train=torch.cat((idx_train, torch.LongTensor([r_s[0]]) ), 0)
        labels= labels.type(torch.LongTensor)
    else:
        labels = torch.zeros((len(features))).type(torch.LongTensor)
        idx_train = torch.LongTensor([0])
    
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    deg = np.diag(adj.toarray().sum(axis=1))
    laplacian = torch.from_numpy((deg - adj.toarray()).astype(np.float32))
    adj_normalized = normalize(sp.csr_matrix(adj) + sp.eye(adj.shape[0]))
    edge_index, edge_attr = from_scipy_sparse_matrix(adj_normalized)

    # adj = sparse_mx_to_torch_sparse_tensor(adj)
    train_mask = torch.zeros(len(features), dtype=torch.bool)
    train_mask[idx_train] = True

    data = Data(
        x=features,  # 节点特征
        edge_index=edge_index,  # 边索引
        edge_attr=edge_attr,  # 边特征（归一化的邻接矩阵权重）
        y=labels,  # 节点标签
        train_mask=train_mask,  # 训练掩码
        # laplacian=laplacian  # 拉普拉斯矩阵（可选）
        ptr=torch.tensor([0, node_ptrs])
    )


    if args.cuda and torch.cuda.is_available():
        model.cuda()
        data = data.cuda()
        features = features.cuda()
        # adj = adj.cuda()
        laplacian =laplacian.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        train_mask = train_mask.cuda()
        edge_index = edge_index.cuda()
        edge_attr = edge_attr.cuda()

    t_total = time.time()
    for epoch in range(args.gcn_epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        # node_embeddings, adj_matrix, timestamps
        # graph mamba

        output = model([data['x']], [[data['edge_index']]], timestamp=1)
        
        # output = model(data)
        # assert not torch.isnan(output).any(), "Model output contains NaN!"
        # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        loss_train = F.nll_loss(F.log_softmax(output, dim=1)[idx_train], labels[idx_train])
        soft_out = torch.unsqueeze(torch.nn.functional.softmax(output,dim=1)[:,1], 1)
        loss_reg = torch.mm(torch.mm(soft_out.T,laplacian), soft_out)
        loss_train += args.gcn_lambda * loss_reg.squeeze()
        loss_train.backward()
        optimizer.step()
