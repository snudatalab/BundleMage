import pickle
import random

import numpy as np
import torch
from sklearn.metrics import ndcg_score

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')


def load_obj(name):
    """
    Load pickle file
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


def set_seed(seed):
    """
    Set random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def spy_sparse2torch_sparse(data):
    """
    Transform scipy sparse tensor to torch sparse tensor
    """
    samples = data.shape[0]
    features = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = torch.LongTensor(np.array([coo_data.row, coo_data.col]))
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), [samples, features])
    return t


def naive_sparse2tensor(data):
    """
    Transform torch sparse tensor to torch dense tensor
    """
    return torch.FloatTensor(data.toarray())


def evaluate_accuracies(pred, ks: list, batch=False):
    """
    Evaluate recalls and ndcgs
    """
    recalls = []
    ndcgs = []
    for k in ks:
        pos_idx = 0
        pred_rank = torch.argsort(pred, dim=1, descending=True)
        pos = (pred_rank == pos_idx).float()
        recall = pos[:, :k].sum()
        if not batch:
            recall = recall/pred.shape[0]
        hit_rank = pos[:, :k].nonzero()[:, 1]
        ndcg = torch.log(torch.full(hit_rank.shape, 2.)).to(DEVICE) / torch.log(2+hit_rank.float())
        ndcg =ndcg.sum()
        if not batch:
            ndcg = ndcg/pred.shape[0]
        recalls.append(recall.item())
        ndcgs.append(ndcg.item())
    return recalls, ndcgs


def evaluate_gen_accruacies(input, target, ks: list):
    """
    Evaluate recalls and ndcgs for bundle generation
    """
    recalls = []
    ndcgs = []
    for k in ks:
        pred = input.topk(k, dim=1, sorted=False)[1]
        row_index = torch.arange(target.size(0))
        target_list = []
        for i in range(k):
            target_list.append(target[row_index, pred[:, i]])
        num_pred = torch.stack(target_list, dim=1).sum(dim=1)
        num_true = target.sum(dim=1)
        recall = (num_pred[num_true > 0] / num_true[num_true > 0]).sum().item()
        recalls.append(recall)
        ndcg = ndcg_score(target.cpu(), input.cpu(), k=k) * input.shape[0]
        ndcgs.append(ndcg)
    return recalls, ndcgs
