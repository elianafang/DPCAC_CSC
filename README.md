# DPCAC: Effective Multi-scale Graph Contrastive Learning via Dual-Perspective Clustering and Adaptive Contrast [🔗](https://www.sciencedirect.com/science/article/pii/S0950705125012882?via%3Dihub)

DPCAC is a self-supervised graph representation learning framework that integrates dual-perspective graph abstraction and stratified similarity-weighted contrastive learning. It is designed to extract expressive node embeddings from unlabeled graph-structured data by combining structural and feature-based semantics with adaptive negative sample differentiation.
<img width="1342" height="475" alt="image" src="https://github.com/user-attachments/assets/5d2d12be-6c35-407a-a6ee-e58edd42ea3f" />

## 🚀 Run
```bash
python main_new39.py --dataset Amazon-Computers --param local:amazon_computers.json --thod 0.7 --fn_thod 0.1
```
## 📦 Datasets
* Amazon-Computers
* Amazon-Photo
* Coauthor-CS
* WikiCS

## 🧩 Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric 2.3.1

## ✍️ Citation
```
@article{li2025dpcac,
  title={DPCAC: Effective Multi-Scale Graph Contrastive Learning via Dual-Perspective Clustering and Adaptive Contrast},
  author={Li, Fangjing and Wang, Zhihai and Wang, Diping and Liu, Haiyang and Ding, Xinxin},
  journal={Knowledge-Based Systems},
  pages={114247},
  year={2025},
  publisher={Elsevier}
}
```
## 📬 Contact

For questions or collaborations, feel free to contact: [eelianafang@gmail.com](eelianafang@gmail.com)
