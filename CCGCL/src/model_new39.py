import os
import sys

import torch
# from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans as KMeans
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from math import exp
from src.functional import cos_sim,RBF_sim


class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr

class Encoder(nn.Module):
    def __init__(self,in_channels,hidden_layer,out_channels,activation):
        super(Encoder,self).__init__()
        self.activation = activation
        self.conv = nn.ModuleList([
            GCNConv(in_channels,hidden_layer),
            GCNConv(hidden_layer,out_channels)
        ])
    def forward(self,x,edge_index):
        for layer in self.conv:
            x = self.activation(layer(x,edge_index))
        return x

class Projection(nn.Sequential):
    def __init__(self,num_hidden,num_proj_hidden):
        super(Projection,self).__init__(
            nn.Linear(num_hidden,num_proj_hidden),
            nn.ELU(),
            nn.Linear(num_proj_hidden,num_hidden)
        )
    def forward(self,x):
        x = super(Projection,self).forward(x)
        return F.normalize(x)

class gCooL(nn.Module):
    def __init__(
            self                            ,
            encoder         : nn.Module     ,
            num_hidden      : int           ,
            num_proj_hidden : int           ,
            tau             : float = 0.4   ,
            num_community   : int   = 10    ,
            lamda           : float = 1.0   ,
            gamma           : float = 8e-5  ,
            alpha_rate      : float = 0.2   ,
            stride          : int   = 500   ,
            similarity      : str   = 'cos',
            thod: float = 0.8,
            aug_node: bool=True,
        ):
        super(gCooL,self).__init__()

        # hyper-parameter
        self.tau = tau
        self.num_community = num_community
        self.lamda = lamda
        self.gamma = gamma
        self.alpha_rate = alpha_rate
        self.stride = stride
        self.thod = thod
        self.aug_node=aug_node

        # similarity measure
        assert similarity in {'cos','RBF'}, 'Unknown similarity measure!'
        self.similarity = similarity

        # backbones
        self.encoder = encoder
        self.center = nn.Parameter(torch.randn(self.num_community,num_hidden,dtype=torch.float32))
        self.proj = Projection(num_hidden,num_proj_hidden)

    def forward(self,x,edge_index):
        return self.encoder(x,edge_index)

    def community_assign(self,h):
        if self.similarity=='cos':
            return cos_sim(h.detach(),F.normalize(self.center),self.tau,norm=True) #这里的self.center是可学习的参数，不是通过计算算出来的
        return RBF_sim(h.detach(),F.normalize(self.center),self.tau,norm=True)

    def node_contrast(self,h1,h2):
        # compute similarity
        s12 = cos_sim(h1,h2,self.tau,norm=False)
        s21 = s12

        # compute InfoNCE
        loss12 = -torch.log(s12.diag())+torch.log(s12.sum(1))
        loss21 = -torch.log(s21.diag())+torch.log(s21.sum(1))
        L_node = (loss12+loss21)/2

        return L_node.mean()

    # def aug_node_embedding(self,h,center,cluster_temp: int=1):
    #     # 计算范数
    #     norm = center.norm(dim=1, keepdim=True)
    #     # 用非常小的值代替零向量的范数
    #     norm = torch.where(norm == 0, torch.tensor(1e-8, device=norm.device), norm)
    #     center = center / norm #对每个聚类中心进行归一化处理，使每个聚类中心的范数为 1
    #     dist = torch.mm(h, center.transpose(0, 1))
    #     # cluster responsibilities via softmax
    #     r = F.softmax(cluster_temp * dist, dim=1)
    #     r = torch.where(r <= self.thod, torch.tensor(0.0, device=r.device), r) #设阈值
    #     # aug_h = torch.sigmoid(r @ center)
    #     aug_h = F.relu(r @ center)
    #     h =h + aug_h
    #     return h

    def node_aug_contrast(self,h1,h2,index1,index2,fn_thod):
        # center1=torch.stack(center1)
        # center2=torch.stack(center2)

        # 生成共现矩阵
        co1 = (index1.unsqueeze(0) == index1.unsqueeze(1)).float()
        co2 = (index2.unsqueeze(0) == index2.unsqueeze(1)).float()

        distb_sim_matrix=co1 + co2  #分配相似度矩阵 强相似2 弱相似1 不相似0
        diag_indices = torch.arange(distb_sim_matrix.size(0)).cuda()
        distb_sim_matrix[distb_sim_matrix == 2] = fn_thod
        distb_sim_matrix[distb_sim_matrix == 0] = 1
        inverse_d = distb_sim_matrix
        inverse_d[diag_indices, diag_indices] = 0
        # # 对反转后的张量按行进行归一化
        # row_sums = inverse_d.sum(dim=1, keepdim=True)
        # d_normalized = inverse_d / row_sums

        # compute similarity
        s12 = cos_sim(h1,h2,self.tau,norm=False)
        s21 = s12

        # off_diag_s12=s12
        # off_diag_s21=s21
        weight_s12 = inverse_d * s12
        weight_s21 = inverse_d * s21

        # compute InfoNCE
        loss12 = -torch.log(s12.diag())+torch.log(s12.diag()+weight_s12.sum(1))
        loss21 = -torch.log(s21.diag())+torch.log(s21.diag()+weight_s21.sum(1))
        L_node = (loss12+loss21)/2

        return L_node.mean()

    def DeCA(self,R,edge_index):
        n = len(R)
        m = n*(n-1)

        # adjacent matrix
        A = torch.zeros(n,n,device=R.device,dtype=torch.float32)
        A[edge_index[0],edge_index[1]] = 1

        # edge density constraint
        DF = R.T@A@R
        return (self.lamda*DF.sum()-(n-1+self.lamda)*DF.trace())/m+(R.T@R).trace()/n/2

    def community_contrast(self,h1,h2,index1,index2,C1,C2, community_strength1,community_strength2):
        # print(communicity_strength)
        # index1, index2, C1, C2 = self.cluster_embedding(h1, h2, R1, R2_prediction, R2_cluster_centers)
        C1,C2 = torch.stack(C1),torch.stack(C2)

        # compute similarity
        #根据设定的相似性度量方式（余弦相似度或 RBF 相似度）计算节点与社区中心之间的相似性矩阵 s_h1_c2 和 s_h2_c1
        if self.similarity=='cos':
            s_h1_c2 = cos_sim(h1,C2.detach(),self.tau,norm=False)
            s_h2_c1 = cos_sim(h2,C1.detach(),self.tau,norm=False)
            ws_h1_c2 = s_h1_c2 + 0.1*community_strength2
            ws_h2_c1 = s_h2_c1 + 0.1* community_strength1
        else:
            #如果使用 RBF 相似度，还会计算重新加权的相似性矩阵 ws_h1_c2 和 ws_h2_c1，其中权重 w1 和 w2 是基于 RBF 核的权重函数
            s_h1_c2 = RBF_sim(h1,C2.detach(),self.tau,norm=False)
            s_h2_c1 = RBF_sim(h2,C1.detach(),self.tau,norm=False)
            h1_extend,h2_extend = torch.stack([C1.detach()]*len(C1)),torch.stack([C2.detach()]*len(C2))
            h1_sub,h2_sub = h1_extend-h1_extend.transpose(0,1),h2_extend-h2_extend.transpose(0,1)
            w1,w2 = torch.exp(-self.gamma*(h1_sub*h1_sub).sum(2)),torch.exp(-self.gamma*(h2_sub*h2_sub).sum(2))
            ws_h1_c2 = s_h1_c2*w2[index2]
            ws_h2_c1 = s_h2_c1*w1[index1]

        # node-community contrast
        # self_s12 和 self_s21 获取每个节点与其对应社区中心的相似性值
        self_s12 = s_h1_c2.gather(1,index2.unsqueeze(-1)).squeeze(-1)
        self_s21 = s_h2_c1.gather(1,index1.unsqueeze(-1)).squeeze(-1)
        loss12 = -torch.log(self_s12)+torch.log( \
            self_s12+ \
            ws_h1_c2.sum(1)-ws_h1_c2.gather(1,index2.unsqueeze(-1)).squeeze(-1)
        )
        loss21 = -torch.log(self_s21)+torch.log( \
            self_s21+ \
            ws_h2_c1.sum(1)-ws_h2_c1.gather(1,index1.unsqueeze(-1)).squeeze(-1)
        )
        L_community = (loss12+loss21)/2

        return L_community.mean()

    def cluster_embedding(self,h1,h2,R1,R2_prediction,R2_cluster_centers):
        # gather communities
        # 通过 R1 和 R2 获取每个节点所属的社区索引 index1 和 index2
        index1 = R1.detach().argmax(dim=1)
        index2 = R2_prediction.long()
        # DeCA通过计算每个社区的节点表示的平均值，来得到每个社区的中心 C1 。C2直接取的kmeans聚类中心。这对应了公式中的社区中心矩阵Φ
        C1 = []
        for i in range(self.num_community):
            h_c1 = h1[index1 == i]
            if h_c1.shape[0] > 0:
                C1.append(h_c1.sum(dim=0) / h_c1.shape[0])
            else:
                C1.append(torch.zeros(h1.shape[1], device=h1.device, dtype=torch.float32))

        C2 = [c for c in R2_cluster_centers]
        return index1,index2,C1,C2

    def kmeans_on_gpu(self,h,device):
        with SuppressOutput():
            prediction, cluster_centers = KMeans(X=h, num_clusters=self.num_community, distance='euclidean',
                                                 device=device)
            prediction = prediction.to(device)
            cluster_centers = cluster_centers.to(device)

        return prediction, cluster_centers

    def kmeans_on_gpu1(self,h,device):
        with SuppressOutput():
            prediction, cluster_centers = KMeans(X=h, num_clusters=self.num_community, distance='euclidean',
                                                 device=device)
            prediction = prediction.to(device)
            cluster_centers = cluster_centers.to(device)

        return prediction, cluster_centers

    def fit(self,epoch,opt,x1,x2,edge_index_1,edge_index_2,fn_thod):
        opt.zero_grad()

        # node contrast
        z1,z2 = self(x1,edge_index_1),self(x2,edge_index_2)
        h1,h2 = self.proj(z1),self.proj(z2)
        # L_node = self.node_contrast(h1,h2)

        # community contrast
        R1= self.community_assign(h1)

        prediction, cluster_centers = self.kmeans_on_gpu(h2,h2.device)

        # km = KMeans(n_clusters=self.num_community, n_init=10).fit(h2.detach().cpu().numpy())
        # prediction = km.predict(h2.detach().cpu().numpy())
        # prediction=torch.tensor(prediction).to(h1.device)
        # cluster_centers = torch.tensor(km.cluster_centers_).to(h1.device)


        index1, index2, C1, C2 = self.cluster_embedding(h1,h2,R1,prediction,cluster_centers)

        if(self.aug_node==True):
            L_node = self.node_aug_contrast(h1,h2, index1, index2, fn_thod)
        else:
            L_node = self.node_contrast(h1, h2)

        # community_strengths = self.community_strength(graph, index1)
        community_strengths1 = self.com_strength(index1,edge_index_1,z1.size(0))#结构聚类的社区强度
        R_aug = R1 + 0.1 * community_strengths1
        DeCA = self.DeCA(R_aug, edge_index_1)

        community_strengths2 = self.com_strength(index2,edge_index_2,z2.size(0)) # kmeans聚类的社区强度
        L_community = self.community_contrast(h1,h2,index1,index2,C1,C2,community_strengths1,community_strengths2)

        # joint objective
        alpha = max(0,1-epoch/self.alpha_rate/self.stride)
        coef = exp(-epoch/self.stride)
        loss = L_node+alpha*(coef*DeCA+(1-coef)*L_community)

        loss.backward()
        opt.step()

        return loss

    def com_strength(self,index,edge_index,num_node):
        # 社区数量k
        k = self.num_community

        # 获取所有边的起点和终点的社区标签
        start_communities = index[edge_index[0]]
        end_communities = index[edge_index[1]]

        # 计算社区对 (start_communities, end_communities) 的边数
        community_pair_counts = torch.stack([start_communities, end_communities], dim=1)
        unique_pairs, counts = community_pair_counts.unique(dim=0, return_counts=True)

        # 创建一个 (k, k) 的矩阵来存储社区边数，数据类型为 long
        community_edge_count = torch.zeros((k, k), dtype=torch.long, device=edge_index.device)

        # 更新社区边数矩阵
        community_edge_count[unique_pairs[:, 0], unique_pairs[:, 1]] = counts
        # print(community_edge_count)
        # 计算每个簇中的节点数
        community_node_count = torch.zeros(k, dtype=torch.long, device=edge_index.device)
        community_node_count = community_node_count.scatter_add(0, index, torch.ones_like(index, dtype=torch.long))
        # print(community_node_count)

        edge_strength=community_edge_count.diag()/edge_index.size(1)
        node_strength=community_node_count/num_node
        com_strength = self.thod *edge_strength + (1-self.thod)*node_strength
        return com_strength
        # # 假设 k 是社区的数量
        # k = torch.max(index).item() + 1
        #
        # # 初始化社区边数矩阵
        # community_edges = torch.zeros((k, k), dtype=torch.int32).to(edge_index.device)
        #
        # # 获取边的起点和终点
        # src, dst = edge_index
        #
        # # 获取起点和终点的社区编号
        # src_community = index[src]
        # dst_community = index[dst]
        #
        # # 使用 scatter_add 将边计数加到社区矩阵中
        # community_edges.index_add_(0, src_community,
        #                            torch.nn.functional.one_hot(dst_community, num_classes=k).sum(dim=0))
        # community_edges.index_add_(1, dst_community,
        #                            torch.nn.functional.one_hot(src_community, num_classes=k).sum(dim=0))
        #
        # # 社区边数矩阵
        # community_edges = community_edges // 2  # 因为每条边会被计数两次
        # print(community_edges)
    def community_strength(self,graph,index):
        # graph = convert_graph_formats(graph, nx.Graph)
        # graph = to_networkx(graph, to_undirected=True)
        coms = {i: int(c) for i, c in enumerate(index)} #构建一个字典coms，将每个节点映射到其所属的社区编号
        inc, deg = {}, {}  # inc记录每个社区的内部边数  deg每个记录社区的总节点度数
        links = graph.size(weight="weight") #同时计算图中总的链接数links，并确保图中有链接
        assert links > 0, "A graph without link has no communities."
        # 遍历图中的每个节点，计算每个社区的节点度数和内部链接数。如果一个链接的两个节点都属于同一个社区，则将该链接计入该社区的内部链接数
        for node in graph:
            try:
                com = coms[node]
                deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight")
                for neighbor, dt in graph[node].items():
                    weight = dt.get("weight", 1)
                    if coms[neighbor] == com:
                        if neighbor == node:
                            inc[com] = inc.get(com, 0.0) + float(weight)
                        else:
                            inc[com] = inc.get(com, 0.0) + float(weight) / 2.0
            except:
                pass
        com_cs = torch.zeros(self.num_community).to(self.center.device)
        for com in range(self.num_community):
            com_cs[com] = ((inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2)
        return com_cs

class LogReg(nn.Module):
    def __init__(self,in_channel:int,num_class:int):
        super(LogReg,self).__init__()
        self.fc = nn.Linear(in_channel,num_class)
        for m in self.modules():
            self.weights_init(m)
    def weights_init(self,m):
        if isinstance(m,nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)
    def forward(self,seq):
        return self.fc(seq)


