import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score,precision_recall_curve,auc, f1_score,cohen_kappa_score
from sklearn.model_selection import train_test_split
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset

def evaluate(y_pred, labels):
    roc_auc = roc_auc_score(labels, y_pred)
    pr,re,_=precision_recall_curve(labels,y_pred,pos_label=1)
    aupr = auc(re, pr)
    y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    recall = recall_score(labels, y_pred)
    precision = precision_score(labels, y_pred)
    acc = accuracy_score(labels, y_pred)
    metrics = {"auc": roc_auc, "recall": recall, "precision": precision, "acc": acc, "aupr": aupr}
    return metrics

def evaluate_multi_class(y_pred, labels):
    # import ipdb; ipdb.set_trace()
    # roc_auc = roc_auc_score(labels, y_pred)
    
    # y_pred = [1 if i > 0.5 else 0 for i in y_pred]
    scores = np.max(y_pred, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    auc = f1_score(labels, y_pred, average='macro')
    aupr = f1_score(labels, y_pred, average='micro')
    kappa = cohen_kappa_score(labels, y_pred)
    recall = recall_score(labels, y_pred, average='macro')
    # precision = precision_score(labels, y_pred)
    acc = accuracy_score(labels, y_pred)
    metrics = {"f1": auc, "recall": recall, "kappa": kappa, "acc": acc}
    return metrics

def eval_loader(model, loader, device, mc=False, graph=None, aug=False):
    if mc:
        e = evaluate_multi_class
    else:
        e = evaluate
    model.eval()
    # model.cuda()
    model = model.to(device)
    Y_pre = []
    Y_true = []
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        h, t, c, d1_embed, d2_embed, c_embed, l = batch
        h, t, c, d1_embed, d2_embed, c_embed, l = h.to(device), t.to(device),c.to(device), d1_embed.to(device), d2_embed.to(device), c_embed.to(device), l.to(device)
        with torch.no_grad():
            pred, _ = model(graph, h, t, c, d1_embed, d2_embed, c_embed, aug=aug)
            Y_pre.extend(list(pred.cpu().detach().numpy()))
            Y_true.extend(list(l.cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/-'.format(str(b_idx+1)))
    return e(np.array(Y_pre), np.array(Y_true))

def save_model(args, model):
    path = '{}/ds_{}_num_{}_dim{}_{}_{}_{}_{}.pt'.format(args.ckpt_dir, args.dataset, args.model, args.batch_size, args.flag, args.llm, args.aug, args.setting)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(args, model):
    path = '{}/ds_{}_num_{}_dim{}_{}_{}_{}_{}.pt'.format(args.ckpt_dir, args.dataset, args.model, args.batch_size, args.flag, args.llm, args.aug, args.setting)
    model.load_state_dict(torch.load(path))
    return model


def log_info(args, info):   
    args.log_file.write('{}\n'.format(info))

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}