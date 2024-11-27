import torch
import torch.nn as nn
import torch.nn.functional as F
from munkres import Munkres
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import Tensor
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv,SGConv,SAGEConv,GATConv,GraphConv,GINConv
from torch_geometric.utils import degree,to_networkx
from torch_scatter import scatter
import networkx as nx
import numpy as np
import random
import scipy.sparse as sp
from sklearn import metrics
from src.kmeans_gpu import kmeans
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from kmeans_pytorch import kmeans as Kmeans


def setup_seed(seed:int=None):
    torch.backends.cudnn.enabled = True
    if seed:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        print('Random seed set to be: %d' % seed)
    else:
        torch.backends.cudnn.benchmark = True

def get_base_model(name:str):
    def gat_wrapper(in_channels,out_channels):
        return GATConv(
            in_channels     = in_channels           ,
            out_channels    = out_channels//4       ,
            heads           = 4
        )
    def gin_wrapper(in_channels,out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels,2*out_channels)   ,
            nn.ELU()                                ,
            nn.Linear(2*out_channels,out_channels)
        )
        return GINConv(mlp)
    base_models = {
        'GCNConv'   : GCNConv                       ,
        'SGConv'    : SGConv                        ,
        'SAGEConv'  : SAGEConv                      ,
        'GATConv'   : gat_wrapper                   ,
        'GraphConv' : GraphConv                     ,
        'GINConv'   : gin_wrapper
    }

    return base_models[name]

def get_activation(name:str):
    activations = {
        'relu'          : F.relu                    ,
        'elu'           : F.elu                     ,
        'prelu'         : torch.nn.PReLU()          ,
        'rrelu'         : F.rrelu                   ,
        'selu'          : F.selu                    ,
        'celu'          : F.celu                    ,
        'leaky_relu'    : F.leaky_relu              ,
        'rrelu'         : torch.nn.RReLU()          ,
        'gelu'          : torch.nn.GELU()           ,
        'softplus'      : F.softplus                ,
        'tanh'          : F.tanh                    ,
        'sigmoid'       : F.sigmoid
    }

    return activations[name]

def compute_pr(edge_index,damp:float=0.85,k:int=10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes,)).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]]/deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')
        x = (1-damp)*x+damp*agg_msg

    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)

def generate_split(num_samples:int,train_ratio:float,val_ratio:float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

def sparse_to_tuple(sparse_mx, insert_batch=False):
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


# Perform train-test split
# Takes in adjacency matrix in sparse format
# Returns: adj_train, train_edges, val_edges, val_edges_false,
# test_edges, test_edges_false
def mask_test_edges(adj, test_frac=.1, val_frac=.05, prevent_disconnect=True, verbose=False):
    # NOTE: Splits are randomized and results might slightly deviate from reported numbers in the paper.
    "from https://github.com/tkipf/gae"

    if verbose == True:
        print('preprocessing...')

    # Remove diagonal elements
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()  # 将邻接矩阵的对角元素置零
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    # 使用NetworkX库从稀疏矩阵创建一个图，并计算原始图的连通分量数
    g = nx.from_scipy_sparse_array(adj)
    orig_num_cc = nx.number_connected_components(g)
    # 获取邻接矩阵的上三角部分，确保每条边只记录一次（防止重复记录）
    adj_triu = sp.triu(adj)  # upper triangular portion of adj matrix
    adj_tuple = sparse_to_tuple(adj_triu)  # (coords, values, shape), edges only 1 way
    edges = adj_tuple[0]  # all edges, listed only once (not 2 ways)
    # edges_all = sparse_to_tuple(adj)[0] # ALL edges (includes both ways)
    # 计算测试集和验证集的边数
    num_test = int(np.floor(edges.shape[0] * test_frac))  # controls how large the test set should be
    num_val = int(np.floor(edges.shape[0] * val_frac))  # controls how alrge the validation set should be

    # 初始化边的集合并随机打乱边列表
    # Store edges in list of ordered tuples (node1, node2) where node1 < node2
    edge_tuples = [(min(edge[0], edge[1]), max(edge[0], edge[1])) for edge in edges]
    all_edge_tuples = set(edge_tuples)
    train_edges = set(edge_tuples)  # initialize train_edges to have all edges
    test_edges = set()
    val_edges = set()

    if verbose == True:
        print('generating test/val sets...')

    # Iterate over shuffled edges, add to train/val sets
    np.random.shuffle(edge_tuples)
    # 迭代处理边，将其分配到训练集、验证集和测试集中
    for edge in edge_tuples:
        # print edge
        node1 = edge[0]
        node2 = edge[1]

        # If removing edge would disconnect a connected component, backtrack and move on
        g.remove_edge(node1, node2)
        if prevent_disconnect == True:
            if nx.number_connected_components(g) > orig_num_cc:
                g.add_edge(node1, node2)
                continue

        # Fill test_edges first
        # 集合中边的数量小于num_test（测试集中需要的边数），则将当前边添加到test_edges中，并从train_edges中移除
        if len(test_edges) < num_test:
            test_edges.add(edge)
            train_edges.remove(edge)

        # Then, fill val_edges
        elif len(val_edges) < num_val:
            val_edges.add(edge)
            train_edges.remove(edge)

        # Both edge lists full --> break loop
        # 如果test_edges和val_edges都已填满，跳出循环
        elif len(test_edges) == num_test and len(val_edges) == num_val:
            break

    if (len(val_edges) < num_val or len(test_edges) < num_test):
        print("WARNING: not enough removable edges to perform full train-test split!")
        print("Num. (test, val) edges requested: (", num_test, ", ", num_val, ")")
        print("Num. (test, val) edges returned: (", len(test_edges), ", ", len(val_edges), ")")

    if prevent_disconnect == True:
        assert nx.number_connected_components(g) == orig_num_cc

    if verbose == True:
        print('creating false test edges...')

    # 生成虚假测试边、验证边和训练边
    # 生成与 num_test 数量相等的虚假测试边，确保这些虚假边不在原始边集中，并且不重复
    test_edges_false = set()
    while len(test_edges_false) < num_test:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge not an actual edge, and not a repeat
        if false_edge in all_edge_tuples:
            continue
        if false_edge in test_edges_false:
            continue

        test_edges_false.add(false_edge)

    if verbose == True:
        print('creating false val edges...')
    # 生成与num_val数量相等的虚假验证边，确保这些虚假边不在原始边集中、虚假测试边集中，并且不重复。
    val_edges_false = set()
    while len(val_edges_false) < num_val:
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in val_edges_false:
            continue

        val_edges_false.add(false_edge)

    if verbose == True:
        print('creating false train edges...')
    # 生成与train_edges数量相等的虚假训练边，确保这些虚假边不在原始边集中、虚假测试边和验证边集中，并且不重复。
    train_edges_false = set()
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue

        false_edge = (min(idx_i, idx_j), max(idx_i, idx_j))

        # Make sure false_edge in not an actual edge, not in test_edges_false,
        # not in val_edges_false, not a repeat
        if false_edge in all_edge_tuples or \
                false_edge in test_edges_false or \
                false_edge in val_edges_false or \
                false_edge in train_edges_false:
            continue

        train_edges_false.add(false_edge)

    if verbose == True:
        print('final checks for disjointness...')

    # 检查边集的相互独立性
    # assert: false_edges are actually false (not in all_edge_tuples)
    assert test_edges_false.isdisjoint(all_edge_tuples)
    assert val_edges_false.isdisjoint(all_edge_tuples)
    assert train_edges_false.isdisjoint(all_edge_tuples)

    # assert: test, val, train false edges disjoint
    assert test_edges_false.isdisjoint(val_edges_false)
    assert test_edges_false.isdisjoint(train_edges_false)
    assert val_edges_false.isdisjoint(train_edges_false)

    # assert: test, val, train positive edges disjoint
    assert val_edges.isdisjoint(train_edges)
    assert test_edges.isdisjoint(train_edges)
    assert val_edges.isdisjoint(test_edges)

    if verbose == True:
        print('creating adj_train...')

    # 创建训练集的邻接矩阵
    # Re-build adj matrix using remaining graph
    adj_train = nx.adjacency_matrix(g)

    # 将边集转换为numpy数组并返回
    # Convert edge-lists to numpy arrays
    train_edges = np.array([list(edge_tuple) for edge_tuple in train_edges])
    train_edges_false = np.array([list(edge_tuple) for edge_tuple in train_edges_false])
    val_edges = np.array([list(edge_tuple) for edge_tuple in val_edges])
    val_edges_false = np.array([list(edge_tuple) for edge_tuple in val_edges_false])
    test_edges = np.array([list(edge_tuple) for edge_tuple in test_edges])
    test_edges_false = np.array([list(edge_tuple) for edge_tuple in test_edges_false])

    if verbose == True:
        print('Done with train-test split!')
        print('')

    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges, train_edges_false, \
           val_edges, val_edges_false, test_edges, test_edges_false

class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper.

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder.
    """
    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        sigmoid: bool = True,
    ) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


def get_roc_score(pos_edge_index, neg_edge_index, z):
    pos_y = z.new_ones(pos_edge_index.size(1))
    neg_y = z.new_zeros(neg_edge_index.size(1))
    y = torch.cat([pos_y, neg_y], dim=0)

    decoder = InnerProductDecoder()
    pos_pred = decoder(z, pos_edge_index, sigmoid=True)
    neg_pred = decoder(z, neg_edge_index, sigmoid=True)
    pred = torch.cat([pos_pred, neg_pred], dim=0)

    y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

    return roc_auc_score(y, pred), average_precision_score(y, pred)

# def get_roc_score(edges_pos, edges_neg, embeddings, adj_sparse):
#     "from https://github.com/tkipf/gae"
#
#     score_matrix = np.dot(embeddings, embeddings.T)
#
#     def sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     # Store positive edge predictions, actual values
#     preds_pos = []
#     pos = []
#     for edge in edges_pos:
#         preds_pos.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
#         pos.append(adj_sparse[edge[0], edge[1]])  # actual value (1 for positive)
#
#     # Store negative edge predictions, actual values
#     preds_neg = []
#     neg = []
#     for edge in edges_neg:
#         preds_neg.append(sigmoid(score_matrix[edge[0], edge[1]]))  # predicted score
#         neg.append(adj_sparse[edge[0], edge[1]])  # actual value (0 for negative)
#
#     # Calculate scores
#     preds_all = np.hstack([preds_pos, preds_neg])
#     labels_all = np.hstack([np.ones(len(preds_pos)), np.zeros(len(preds_neg))])
#
#     # print(preds_all, labels_all )
#
#     roc_score = roc_auc_score(labels_all, preds_all)
#     ap_score = average_precision_score(labels_all, preds_all)
#     return roc_score, ap_score
def cluster_acc(y_true, y_pred):
    """
    calculate clustering acc and f1-score
    Args:
        y_true: the ground truth
        y_pred: the clustering id

    Returns: acc and f1-score
    """
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    numclass2 = len(l2)
    if num_class1 != numclass2:
        print('error')
        return
    cost = np.zeros((num_class1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average='macro')
    return acc, f1_macro

def eva(y_true, y_pred, show_details=True):
    """
    evaluate the clustering performance
    Args:
        y_true: the ground truth
        y_pred: the predicted label
        show_details: if print the details
    Returns: None
    """
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    if show_details:
        print(':acc {:.4f}'.format(acc), ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari),
              ', f1 {:.4f}'.format(f1))
    return acc, nmi, ari, f1

def clustering(feature, true_labels, cluster_num):
    predict_labels, cluster_centers = Kmeans(X=feature, num_clusters=cluster_num, distance='euclidean',
                                         device='cuda')
    predict_labels = predict_labels.detach().cpu().numpy()

    # km = KMeans(n_clusters=cluster_num, n_init=10).fit(feature.detach().cpu().numpy())
    # predict_labels = km.predict(feature.detach().cpu().numpy())

    # predict_labels, _ = kmeans(X=feature, num_clusters=cluster_num, distance="euclidean", device="cuda")
    acc, nmi, ari, f1 = eva(true_labels.detach().cpu().numpy(), predict_labels, show_details=False)
    return round(100 * acc, 2), round(100 * nmi, 2), round(100 * ari, 2), round(100 * f1, 2), predict_labels

