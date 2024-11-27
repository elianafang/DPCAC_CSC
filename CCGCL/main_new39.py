
from collections import defaultdict

import torch
import argparse
import csv
import os
import sys
import os.path as osp
from torch_geometric.data import Data

# import nni
# from torch_geometric.utils import degree,to_undirected
from sp import SimpleParam
from src.model_new39 import *
from src.functional import *
from src.eval_new import *
from src.utils import *
from src.dataset import get_dataset
from time import time
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# randomly drop edges by weights
def drop_edge(idx):
    global drop_weights

    assert param['drop_scheme'] in ['degree','evc','pr'], 'Unimplemented drop scheme!'
    return drop_edge_weighted(data.edge_index,drop_weights,p=param[f'drop_edge_rate_{idx}'],threshold=0.7)

# training per epoch
def train_epoch(epoch):
    net.train()

    # drop edges
    edge_index_1,edge_index_2 = drop_edge(1),drop_edge(2)

    # drop features
    x1 = drop_feature_weighted_2(data.x,feature_weights,param['drop_feature_rate_1'])
    x2 = drop_feature_weighted_2(data.x,feature_weights,param['drop_feature_rate_2'])

    # # 由x1和edge_index1生成networkx图对象
    # data1 = Data(x=x1, edge_index=edge_index_1)
    # graph = to_networkx(data1, to_undirected=False)

    # contrastive training
    loss = net.fit(epoch,opt,x1,x2,edge_index_1,edge_index_2,param['fn_thod'])
    return loss.item()

# testing
def test(device):
    net.eval()
    global split

    # encoding
    with torch.no_grad():
        z = net(data.x,data.edge_index)

    cluster_acc, nmi, ari, f1, predict_labels = clustering(z, data.y, param['num_community'])

    # classifier on generated features
    if args.dataset=='WikiCS':
        micro_f1,macro_f1,acc,recall,precision = [],[],[],[],[]
        for i in range(20):
            result = log_regression(z,data.y,dataset,1000,device,split=f'wikics:{i}')
            micro_f1.append(result["Micro-F1"])
            macro_f1.append(result["Macro-F1"])
            acc.append(result["Acc"])
            precision.append(result["Precision"])
            recall.append(result["Recall"])
        micro_f1 = sum(micro_f1)/len(micro_f1)
        macro_f1 = sum(macro_f1)/len(macro_f1)
        acc = sum(acc) / len(acc)
        precision = sum(precision) / len(precision)
        recall = sum(recall) / len(recall)
    else:
        result = log_regression(z,data.y,dataset,5000,device,split='rand:0.1',preload_split=split)
        micro_f1 = result["Micro-F1"]
        macro_f1 = result["Macro-F1"]
        acc = result["Acc"]
        precision = result["Precision"]
        recall = result["Recall"]

    return micro_f1,macro_f1, acc, precision, recall, cluster_acc, nmi, ari

if __name__ == '__main__':
    # hyper-parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, default='Amazon-Photo')
    parser.add_argument('--param', type=str, default='local:amazon_photo.json')
    parser.add_argument('--patience', type=int, default=1000)
    # parser.add_argument('--fn_thod', type=float, default=0.5,help="false-negative node thod")
    default_param = {
        'learning_rate'         : 0.01      ,
        'num_hidden'            : 384       ,
        'num_proj_hidden'       : 384       ,
        'activation'            : 'prelu'   ,
        'base_model'            : 'GCNConv' ,
        'drop_edge_rate_1'      : 0.2       ,
        'drop_edge_rate_2'      : 0.2       ,
        'drop_feature_rate_1'   : 0.1       ,
        'drop_feature_rate_2'   : 0.1       ,
        'tau'                   : 0.4       ,
        'num_epochs'            : 1000      ,
        'weight_decay'          : 1e-5,
        'aug_node': True,
        'thod': 0.5,
        'fn_thod': 0.5,
    }
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}',type=type(default_param[key]),nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param,preprocess='nni')

    # merge cli arguments and parsed parameters
    # 遍历所有参数键，如果命令行参数中有对应的值，就用命令行参数的值覆盖解析的参数文件中的值
    for key in param_keys:
        if getattr(args, key) is not None:  # 检查是否为 None
            param[key] = getattr(args, key)
    # param['num_epochs'] = 100
    best_dict = defaultdict(list)  # 创建一个默认字典 best_dict，用于存储每次实验的最佳结果
    all_start=time()
    for seed in [39789,39788,39790]:
        # set random seed and computing device
        setup_seed(seed)
        device = torch.device(args.device)

        # load dataset
        path = osp.expanduser('~/datasets')
        path = osp.join(path, args.dataset)
        dataset = get_dataset(path,args.dataset)
        data = dataset[0].to(device)

        # generate split
        split = generate_split(data.num_nodes,train_ratio=0.1,val_ratio=0.1)

        # initiate models
        encoder = Encoder(
            dataset.num_features    ,
            param['hidden_layer']   ,
            param['num_hidden']     ,
            get_activation(param['activation'])
        ).to(device)
        net = gCooL(
            encoder         = encoder                   ,
            num_hidden      = param['num_hidden']       ,
            num_proj_hidden = param['num_proj_hidden']  ,
            tau             = param['tau']              ,
            num_community   = param['num_community']    ,
            lamda           = param['lamda']            ,
            gamma           = param['gamma']            ,
            stride          = param['stride'],

            aug_node=param['aug_node'],
            thod= param['thod']
        ).to(device)
        opt = torch.optim.Adam(net.parameters(),lr=param['learning_rate'],weight_decay=param['weight_decay'])

        # weights for dropping edges
        if param['drop_scheme'] == 'degree':
            drop_weights = degree_drop_weights(data.edge_index).to(device)
        elif param['drop_scheme'] == 'pr':
            drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
        elif param['drop_scheme'] == 'evc':
            drop_weights = evc_drop_weights(data).to(device)
        else:
            drop_weights = None

        # weights for dropping features
        if param['drop_scheme'] == 'degree':
            edge_index_ = to_undirected(data.edge_index)
            node_deg = degree(edge_index_[1])
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
        elif param['drop_scheme'] == 'pr':
            node_pr = compute_pr(data.edge_index)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
        elif param['drop_scheme'] == 'evc':
            node_evc = eigenvector_centrality(data)
            if args.dataset == 'WikiCS':
                feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
            else:
                feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = torch.ones((data.x.size(1),)).to(device)

        patience_counter=0

        # training and testing
        start = time()
        best_epoch=0
        best_acc, best_precision, best_recall, best_micro_f1, best_macro_f1 = 0, 0, 0, 0, 0
        best_cluster_acc, best_nmi, best_ari = 0, 0, 0
        for epoch in range(param['num_epochs']+1):
            if patience_counter >= args.patience:
                best_epoch=epoch
                break
            loss = train_epoch(epoch)
            if epoch%10==0:
                print(f'Training epoch: {epoch:04d}, loss = {loss:.4f}')

            if epoch%100==0:
                print()
                print(f'================ Test epoch: {epoch:04d} ================')
                micro_f1,macro_f1, acc, precision, recall,cluster_acc, nmi, ari = test(device)
                print(f'acc = {acc}')
                print(f'precision = {precision}')
                print(f'recall = {recall}')
                print(f'Micro-F1 = {micro_f1}')
                print(f'Macro-F1 = {macro_f1}')
                print(f'cluster_acc = {cluster_acc}')
                print(f'nmi = {nmi}')
                print(f'ari = {ari}')
                print()
                print()

                if acc < best_acc and precision<best_precision and recall<best_recall and micro_f1<best_micro_f1 and macro_f1<best_macro_f1:
                    patience_counter +=100
                else:
                    patience_counter = 0
                    best_epoch = epoch
                    best_acc = max(best_acc, acc)
                    best_precision = max(best_precision, precision)
                    best_recall = max(best_recall, recall)
                    best_micro_f1 = max(best_micro_f1, micro_f1)
                    best_macro_f1 = max(best_macro_f1, macro_f1)

                    best_cluster_acc = max(best_cluster_acc, cluster_acc)
                    best_nmi = max(best_nmi, nmi)
                    best_ari = max(best_ari, ari)

        end = time()
        # 定义并赋值字典来存储本次seed最优结果
        temp_dict = {
            'best_epoch':best_epoch,
            'time': (end-start)/60,
            'best_acc': best_acc,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_micro_f1': best_micro_f1,
            'best_macro_f1': best_macro_f1,

            'best_cluster_acc': best_cluster_acc,
            'best_nmi': best_nmi,
            'best_ari': best_ari,
        }
        for k, v in temp_dict.items():
            best_dict[k].append(v)

        csv_filename = "results_detail.csv"
        if not os.path.exists(csv_filename):
            with open(csv_filename, "w+", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(
                    ["file", "dataset_name", "lr", "thod","seed","fn_thod", "mean_acc", "precision", "recall", "micro_f1", "macro_f1", 'cluster_acc', 'nmi', 'ari',
                     "time(min)", "total_time(min)"])
        with open("results_detail.csv", "a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            writer.writerow(["main_new39",
                             args.dataset,
                             param['learning_rate'],
                             param['thod'],
                             seed,
                             param['fn_thod'],
                             f"{100 * best_acc:.4f}%",
                             f"{100 * best_precision:.4f}% ",
                             f"{100 * best_recall:.4f}%",
                             f"{100 * best_micro_f1:.4f}%",
                             f"{100 * best_macro_f1:.4f}%",

                             f"{100 * best_cluster_acc:.4f}%",
                             f"{100 * best_nmi:.4f}%",
                             f"{100 * best_ari:.4f}%",
                             ])

    for metric in ['acc', 'precision', 'recall', 'micro_f1', 'macro_f1', 'cluster_acc', 'nmi', 'ari']:
        if f'best_{metric}' in best_dict:
            boots_series = sns.algorithms.bootstrap(best_dict[f'best_{metric}'], func=np.mean, n_boot=1000)
            best_dict[f'best_{metric}_ci'] = np.max(
                np.abs(sns.utils.ci(boots_series, 95) - np.mean(best_dict[f'best_{metric}']))
            )
    for k, v in best_dict.items():
        if '_ci' not in k and k != 'best_epoch':
            best_dict[k] = np.mean(best_dict[k])
    print(f"mean_acc: {100 * best_dict['best_acc']:.4f}% +- {100 * best_dict['best_acc_ci']:.2f}%")
    print(f"mean_precision: {100 * best_dict['best_precision']:.4f}% +- {100 * best_dict['best_precision_ci']:.2f}%")
    print(f"mean_recall: {100 * best_dict['best_recall']:.4f}% +- {100 * best_dict['best_recall_ci']:.2f}%")
    print(f"mean_Micro-F1:{100 * best_dict['best_micro_f1']:.4f}% +- {100 * best_dict['best_micro_f1_ci']:.2f}%")
    print(f"mean_Macro-F1: {100 * best_dict['best_macro_f1']:.4f}% +- {100 * best_dict['best_macro_f1_ci']:.2f}%")
    print(f"{best_dict['time']}min")

    print(f"mean_cluster_acc: {best_dict['best_cluster_acc']:.4f}% +- {best_dict['best_cluster_acc_ci']:.2f}%")
    print(f"mean_nmi: {best_dict['best_nmi']:.4f}% +- {best_dict['best_nmi_ci']:.2f}%")
    print(f"mean_ari: {best_dict['best_ari']:.4f}% +- {best_dict['best_ari_ci']:.2f}%")

    # time usage
    all_end=time()
    print(f'Finished in {(all_end-all_start)/60:.1f} min')
    # print(f'best_acc = {best_acc}')
    # print(f'best_precision = {best_precision}')
    # print(f'best_recall = {best_recall}')
    # print(f'best_Micro-F1 = {best_micro_f1}')
    # print(f'best_Macro-F1 = {best_macro_f1}')

    # 检查 CSV 文件是否存在，如果不存在则创建
    csv_filename = "results.csv"
    if not os.path.exists(csv_filename):
        with open(csv_filename, "w+", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["file","dataset_name","lr","thod","seed","fn_thod","mean_acc","precision","recall","micro_f1","macro_f1", 'cluster_acc', 'nmi', 'ari',"time(min)","total_time(min)"])
    with open("results.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(["main_new39",
                         args.dataset,
                         param['learning_rate'],
                         param['thod'],
                         "avg",
                         param['fn_thod'],
                         f"{100 * best_dict['best_acc']:.4f}% +- {100 * best_dict['best_acc_ci']:.2f}%",
                         f"{100 * best_dict['best_precision']:.4f}% +- {100 * best_dict['best_precision_ci']:.2f}%",
                         f"{100 * best_dict['best_recall']:.4f}% +- {100 * best_dict['best_recall_ci']:.2f}%",
                         f"{100 * best_dict['best_micro_f1']:.4f}% +- {100 * best_dict['best_micro_f1_ci']:.2f}%",
                         f"{100 * best_dict['best_macro_f1']:.4f}% +- {100 * best_dict['best_macro_f1_ci']:.2f}%",

                         f"{best_dict['best_cluster_acc']:.4f}% +- {best_dict['best_cluster_acc_ci']:.2f}%",
                         f"{best_dict['best_nmi']:.4f}% +- {best_dict['best_nmi_ci']:.2f}%",
                         f"{best_dict['best_ari']:.4f}% +- {best_dict['best_ari_ci']:.2f}%",

                         f"{best_dict['time']:.1f}",
                         f"{(all_end - all_start) / 60:.1f}"
                         ])