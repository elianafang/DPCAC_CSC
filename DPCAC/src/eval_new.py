import torch
from torch.optim import Adam
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from src.model import LogReg, Decoder


def get_idx_split(dataset,split,preload_split):
    if split[:4]=='rand':
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes*train_ratio)
        indices = torch.randperm(num_nodes)
        return {
            'train' : indices[:train_size]              ,
            'val'   : indices[train_size:2*train_size]  ,
            'test'  : indices[2*train_size:]
        }
    elif split=='ogb':
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        split_idx = int(split.split(':')[1])
        return {
            'train' : dataset[0].train_mask[:,split_idx]    ,
            'test'  : dataset[0].test_mask                  ,
            'val'   : dataset[0].val_mask[:,split_idx]
        }
    elif split=='preloaded':
        assert preload_split, 'preloaded_split not found!'
        train_mask,test_mask,val_mask = preload_split
        return {
            'train' : train_mask    ,
            'test'  : test_mask     ,
            'val'   : val_mask
        }

    return None

def f1(y_true,y_pred,avg:str):
    return f1_score(y_true,y_pred,average=avg)

def evaluate(res):
    y_true=res['y_true'].view(-1).cpu()
    y_pred=res['y_pred'].argmax(-1).cpu()
    acc = y_pred.eq(y_true).sum().item() / len(y_true)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    return {
        "acc":acc,
        "precision":precision,
        "recall": recall,
        "Micro-F1":f1(y_true, y_pred,avg="micro"),
        "Macro-F1":f1(y_true, y_pred,avg="macro")
    }

def evaluate_linkpre(res):
    y_true=res['y_true'].view(-1).cpu().numpy()
    y_pred=res['y_pred'].detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    return {
        "auc":auc,
        "ap":ap,
    }

def log_regression(z,y,dataset,num_epochs:int,device,split="rand:0.1",preload_split=None):
    # classifier settings
    num_hidden = z.size(1)
    num_classes = y.max().item()+1
    y = y.view(-1)
    classifier = LogReg(num_hidden,num_classes).to(device)
    optimizer = Adam(classifier.parameters(),lr=0.01,weight_decay=1e-5)

    # prepare splits
    split = get_idx_split(dataset,split,preload_split)
    split = {k: v.to(device) for k, v in split.items()}

    # obtain test results
    best_acc,best_precision,best_recall,best_micro_f1,best_macro_f1 = 0,0,0,0,0

    for epoch in range(num_epochs+1):
        classifier.train()

        optimizer.zero_grad()
        output = classifier(z[split['train']])
        loss = F.cross_entropy(output,y[split['train']])
        loss.backward()
        optimizer.step()

        # update test results
        if epoch%5==0:
            classifier.eval()
            result = evaluate({
                'y_true': y[split['test']].view(-1,1),
                'y_pred': classifier(z[split['test']])
            })
            best_acc = max(best_acc,result["acc"])
            best_precision = max(best_precision,result["precision"])
            best_recall = max(best_recall,result["recall"])
            best_micro_f1 = max(best_micro_f1,result["Micro-F1"])
            best_macro_f1 = max(best_macro_f1,result["Macro-F1"])

    return {"Micro-F1":best_micro_f1,
            "Macro-F1":best_macro_f1,
            "Acc":best_acc,
            "Precision":best_precision,
            "Recall":best_recall,
            }


def mlp(z,y,dataset,num_epochs:int,device,train_pairs,test_pairs,train_labels,test_labels):
    # classifier settings
    dim_in = z.size(1)
    num_classes = y.max().item()+1
    y = y.view(-1)
    classifier= Decoder(dim_in).to(device)
    # classifier = LogReg(num_hidden,num_classes).to(device)
    optimizer = Adam(classifier.parameters(),lr=0.01,weight_decay=1e-5)

    # # prepare splits
    # split = get_idx_split(dataset,split,preload_split)
    # split = {k: v.to(device) for k, v in split.items()}

    # obtain test results
    best_auc,best_ap = 0,0

    for epoch in range(num_epochs+1):
        classifier.train()

        optimizer.zero_grad()
        output_train = classifier(z[train_pairs[0]], z[train_pairs[1]]).cpu()
        loss = F.binary_cross_entropy_with_logits(output_train, train_labels)
        # output = classifier(z[split['train']])
        # loss = F.cross_entropy(output,y[split['train']])
        loss.backward()
        optimizer.step()

        # update test results
        if epoch%5==0:
            classifier.eval()
            result = evaluate_linkpre({
                'y_true': test_labels.view(-1,1),
                'y_pred': classifier(z[test_pairs[0]], z[test_pairs[1]])
            })
            best_auc = max(best_auc,result["auc"])
            best_ap = max(best_ap,result["ap"])

    return {
            "auc":best_auc,
            "ap":best_ap,
            }

