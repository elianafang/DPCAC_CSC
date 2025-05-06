import os.path as osp
from torch_geometric.datasets import WikiCS, Coauthor, Amazon, Planetoid, WebKB
import torch_geometric.transforms as T

# load dataset
def get_dataset(path:str,name:str):
    root_path = osp.expanduser('~/datasets')

    if name=='Coauthor-CS':
        return Coauthor(root="D:/PostgraduateCode/data/coauther",name='cs',transform=T.NormalizeFeatures())
        # return Coauthor(root="/root/autodl-tmp/Datasets/coauther",name='cs',transform=T.NormalizeFeatures())

    elif name=='WikiCS':
        return WikiCS(root="D:/PostgraduateCode/data/WikiCS")
        # return WikiCS(root="/root/autodl-tmp/Datasets/WikiCS")

    elif name=='Amazon-Computers':
        return Amazon(root="D:/PostgraduateCode/data/Amazon",name='computers',transform=T.NormalizeFeatures())
        # return Amazon(root="/root/autodl-tmp/Datasets/Amazon",name='computers',transform=T.NormalizeFeatures())

    elif name=='Amazon-Photo':
        return Amazon(root="D:/PostgraduateCode/data/Amazon",name='photo',transform=T.NormalizeFeatures())
        # return Amazon(root="/root/autodl-tmp/Datasets/Amazon",name='photo',transform=T.NormalizeFeatures())
    elif name =='Cora':
        return Planetoid("D:/PostgraduateCode/data/", name,transform=T.NormalizeFeatures())
        # return Planetoid("/root/autodl-tmp/Datasets/", name,transform=T.NormalizeFeatures())
    elif name in ['cornell', 'texas', 'wisconsin', 'washington']:
        dataset = WebKB("/root/autodl-fs/GraphDatasets/Homogeneous/webkb", name)

    assert False, 'Unknown dataset!'
