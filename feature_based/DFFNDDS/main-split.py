import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sklearn.metrics as metrics
from dataset import SynergyEncoderDataset
from sklearn.model_selection import  ShuffleSplit, train_test_split
from transformers import AdamW, get_linear_schedule_with_warmup
from dgllife.utils import EarlyStopping
import argparse
import os
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score, roc_auc_score
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
import torch.nn.functional as F

from prettytable import PrettyTable
from typing import ClassVar, Iterable, Mapping, Optional, Sequence, Tuple, Union
from collections import defaultdict
import pandas as pd
import json
from model_h import MultiViewNet
from data_split_standard import *


def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()
    loss = (p_loss + q_loss) / 2
    return loss


def kl_loss_c(q, p):
    kl_pq = F.kl_div(torch.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    kl_qp = F.kl_div(torch.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    sym_kl_div = kl_pq + kl_qp
    return sym_kl_div


def alignment(x, y, alpha=2):
    x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def read_data_file(data_file):
    smiles_1 = []
    smiles_2 = []
    Y = []
    context = []

    with open(data_file, 'r') as f:
        all_lines = f.readlines()
        for line in all_lines[1:]:
            row = line.rstrip().split(',')[1:]
            smiles_1.append(row[0])
            smiles_2.append(row[1])
            context.append((row[2]))
            Y.append(int(row[3]))

    return smiles_2, smiles_1, context, Y


# 定义数据加载器的函数
def define_dataloader(smiles_1_train, smiles_2_train, context_train,
                      LLMsmiles_1_train, LLMsmiles_2_train, LLMcontext_train, Y_train,
                      smiles_1_valid, smiles_2_valid, context_valid, LLMsmiles_1_valid, LLMsmiles_2_valid, LLMcontext_valid, Y_valid,
                      smiles_1_test, smiles_2_test, context_test, LLMsmiles_1_test, LLMsmiles_2_test, LLMcontext_test, Y_test,
                      maxCompoundLen, batch_size):

    smiles_1_train = np.array(smiles_1_train)
    smiles_2_train = np.array(smiles_2_train)
    context_train = np.array(context_train)
    LLMsmiles_1_train = np.array(LLMsmiles_1_train)
    LLMsmiles_2_train = np.array(LLMsmiles_2_train)
    LLMcontext_train = np.array(LLMcontext_train)
    Y_train = np.array(Y_train)

    smiles_1_valid = np.array(smiles_1_valid)
    smiles_2_valid = np.array(smiles_2_valid)
    context_valid = np.array(context_valid)
    LLMsmiles_1_valid = np.array(LLMsmiles_1_valid)
    LLMsmiles_2_valid = np.array(LLMsmiles_2_valid)
    LLMcontext_valid = np.array(LLMcontext_valid)
    Y_valid = np.array(Y_valid)

    smiles_1_test = np.array(smiles_1_test)
    smiles_2_test = np.array(smiles_2_test)
    context_test = np.array(context_test)
    LLMsmiles_1_test = np.array(LLMsmiles_1_test)
    LLMsmiles_2_test = np.array(LLMsmiles_2_test)
    LLMcontext_test = np.array(LLMcontext_test)
    Y_test = np.array(Y_test)

    # 注意：这里用到了全局 device（与你原逻辑一致）
    train_dataset = SynergyEncoderDataset(
        smiles_1_train, smiles_2_train,
        LLMsmiles_1_train, LLMsmiles_2_train,
        Y_train, context_train, LLMcontext_train,
        maxCompoundLen, device=device
    )

    valid_dataset = SynergyEncoderDataset(
        smiles_1_valid, smiles_2_valid,
        LLMsmiles_1_valid, LLMsmiles_2_valid,
        Y_valid, context_valid, LLMcontext_valid,
        maxCompoundLen, device=device
    )

    test_dataset = SynergyEncoderDataset(
        smiles_1_test, smiles_2_test,
        LLMsmiles_1_test, LLMsmiles_2_test,
        Y_test, context_test, LLMcontext_test,
        maxCompoundLen, device=device
    )

    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validLoader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainLoader, validLoader, testLoader


def validate_new(valid_loader, model):
    model.eval()
    preds = torch.Tensor()
    trues = torch.Tensor()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            compounds_1, compounds_2, Y, context, fp1, fp2, LLMdruga, LLMdrugb, LLMcell = batch
            compounds_1 = compounds_1.to(device)
            compounds_2 = compounds_2.to(device)
            Y = Y.to(device)
            context = context.to(device)
            fp1 = fp1.to(device)
            fp2 = fp2.to(device)
            LLMdruga = LLMdruga.to(device)
            LLMdrugb = LLMdrugb.to(device)
            LLMcell = LLMcell.to(device)

            pre_synergy, _ ,_ = model(compounds_1, compounds_2, context, fp1, fp2, LLMdruga, LLMdrugb, LLMcell)
            # 建议指定 dim
            pre_synergy = torch.softmax(pre_synergy, dim=-1)[:, 1]

            preds = torch.cat((preds, pre_synergy.cpu()), 0)
            trues = torch.cat((trues, Y.view(-1, 1).cpu()), 0)

        y_pred = np.array(preds) > 0.5
        accuracy = accuracy_score(trues, y_pred)
        BACC = balanced_accuracy_score(trues, y_pred)
        roc_auc = roc_auc_score(trues, preds)
        ACC = accuracy_score(trues, y_pred)
        F1 = f1_score(trues, y_pred, average='binary')
        Prec = precision_score(trues, y_pred, average='binary')
        Rec = recall_score(trues, y_pred, average='binary')
        kappa = cohen_kappa_score(trues, y_pred)
        mcc = matthews_corrcoef(trues, y_pred)
        ap = average_precision_score(trues, preds)

        return accuracy, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap


def train(train_loader, model, epoch, optimizer, device, scheduler, print_freq=200):
    model.train()
    cross_entropy_loss = nn.CrossEntropyLoss()
    losses = AverageMeter()

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        compounds_1, compounds_2, synergyScores, context, fp1, fp2, LLMdruga, LLMdrugb, LLMcell = batch
        compounds_1 = compounds_1.to(device)
        compounds_2 = compounds_2.to(device)
        synergyScores = synergyScores.to(device)
        context = context.to(device)
        fp1 = fp1.to(device)
        fp2 = fp2.to(device)
        LLMdruga = LLMdruga.to(device)
        LLMdrugb = LLMdrugb.to(device)
        LLMcell = LLMcell.to(device)

        pre_synergy, llm_synergy ,contrastive_loss= model(compounds_1, compounds_2, context, fp1, fp2, LLMdruga, LLMdrugb, LLMcell)
        pre_synergy2, llm_synergy2 ,contrastive_loss2= model(compounds_1, compounds_2, context, fp1, fp2, LLMdruga, LLMdrugb, LLMcell)

        ce_loss = 0.5 * (
            cross_entropy_loss(pre_synergy, synergyScores.squeeze(1)) +
            cross_entropy_loss(pre_synergy2, synergyScores.squeeze(1))
        )
        kl_loss = 0.5*(compute_kl_loss(pre_synergy, llm_synergy)+compute_kl_loss(pre_synergy2, llm_synergy2))

        α = 5
        loss = ce_loss + α * kl_loss  + contrastive_loss
        # + kl_loss 
        # (需要就加回来)

        losses.update(loss.item(), len(compounds_1))
        loss.backward()

        # 你原来是先 scheduler.step() 再 optimizer.step()，保持不动
        scheduler.step()
        if np.isnan(loss.item()):
            raise Exception("Training model diverges.")
        optimizer.step()

        if i % print_freq == 0:
            log_str = 'TRAIN -> Epoch{epoch}: \tIter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch=epoch, iter=i, loss=losses
            )
            print(log_str)


# =========================
# 新增：汇总 5 个 seed 的结果
# =========================
def summarize_results(all_results):
    metrics = ['ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'kappa', 'ap']

    t = PrettyTable(['seed'] + metrics)
    t.float_format = '.4'
    for r in all_results:
        t.add_row([r['seed']] + [r[m] for m in metrics])
    print("\nPer-seed TEST metrics:")
    print(t)

    arr = {m: np.array([r[m] for r in all_results], dtype=float) for m in metrics}
    mean_row = ['mean'] + [arr[m].mean() for m in metrics]
    std_row = ['std'] + [arr[m].std(ddof=1) if len(all_results) > 1 else 0.0 for m in metrics]

    t2 = PrettyTable(['stat'] + metrics)
    t2.float_format = '.4'
    t2.add_row(mean_row)
    t2.add_row(std_row)
    print("\nMean / Std across seeds:")
    print(t2)


# =========================
# 改造：每个 seed 单独跑一次
# =========================
def run_expriments(device, seed):
    set_random_seed(seed)

    n_epochs = 200

    # 读取 CSV 文件v1/
    # train_df = pd.read_csv('v1/inductive/v1-train_fake.csv')
    # valid_df = pd.read_csv('v1/inductive/v1-valid.csv')
    # test_df = pd.read_csv('v1/inductive/v1-test.csv')
    train_df = pd.read_csv('MyDataset/new-cell/db-train.csv')
    valid_df = pd.read_csv('MyDataset/new-cell/db-valid.csv')
    test_df = pd.read_csv('MyDataset/new-cell/db-test.csv')
    drug_data = pd.read_csv('MyDataset/merged_data.csv')
    cell_data = pd.read_csv('MyDataset/cell_map.csv')

    name_to_id = dict(zip(drug_data['name'], drug_data['index']))
    name_to_contex = dict(zip(cell_data['name'], cell_data['context']))

    # 读取大模型映射
    LLMdrug_data = pd.read_csv('MyDataset/LLM_data/drug_map.csv')
    LLMcell_data = pd.read_csv('MyDataset/LLM_data/cell_map.csv')
    name_to_LLMid = dict(zip(LLMdrug_data['name'], LLMdrug_data['index']))
    name_to_LLMcontex = dict(zip(LLMcell_data['name'], LLMcell_data['index']))

    # train
    smiles_1_train = train_df['drug_a'].map(name_to_id).fillna(2578).astype(int)
    smiles_2_train = train_df['drug_b'].map(name_to_id).fillna(2578).astype(int)
    context_train = train_df['cell'].map(name_to_contex)
    LLMsmiles_1_train = train_df['drug_a'].map(name_to_LLMid)
    LLMsmiles_2_train = train_df['drug_b'].map(name_to_LLMid)
    LLMcontext_train = train_df['cell'].map(name_to_LLMcontex)
    Y_train = train_df['synergy'].values

    # valid
    smiles_1_valid = valid_df['drug_a'].map(name_to_id).fillna(2578).astype(int)
    smiles_2_valid = valid_df['drug_b'].map(name_to_id).fillna(2578).astype(int)
    context_valid = valid_df['cell'].map(name_to_contex)
    LLMsmiles_1_valid = valid_df['drug_a'].map(name_to_LLMid)
    LLMsmiles_2_valid = valid_df['drug_b'].map(name_to_LLMid)
    LLMcontext_valid = valid_df['cell'].map(name_to_LLMcontex)
    Y_valid = valid_df['synergy'].values

    # test
    smiles_1_test = test_df['drug_a'].map(name_to_id).fillna(2578).astype(int)
    smiles_2_test = test_df['drug_b'].map(name_to_id).fillna(2578).astype(int)
    context_test = test_df['cell'].map(name_to_contex)
    LLMsmiles_1_test = test_df['drug_a'].map(name_to_LLMid)
    LLMsmiles_2_test = test_df['drug_b'].map(name_to_LLMid)
    LLMcontext_test = test_df['cell'].map(name_to_LLMcontex)
    Y_test = test_df['synergy'].values

    trainLoader, validLoader, testLoader = define_dataloader(
        smiles_1_train, smiles_2_train, context_train,
        LLMsmiles_1_train, LLMsmiles_2_train, LLMcontext_train, Y_train,
        smiles_1_valid, smiles_2_valid, context_valid,
        LLMsmiles_1_valid, LLMsmiles_2_valid, LLMcontext_valid, Y_valid,
        smiles_1_test, smiles_2_test, context_test,
        LLMsmiles_1_test, LLMsmiles_2_test, LLMcontext_test, Y_test,
        128, 256
    )

    model = MultiViewNet()
    model.to(device)

    # 关键：不同 seed 用不同文件名，避免互相覆盖
    stopper = EarlyStopping(mode='higher', filename=f'8%-mainsplit-attention-comb-seed{seed}-v1-laco', patience=50)

    lr = 1e-3
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=n_epochs, steps_per_epoch=len(trainLoader)
    )

    best_epoch = -1
    for epochind in range(n_epochs):
        epoch_start_time = time.time()
        train(trainLoader, model, epochind, optimizer, device, scheduler)
        accuracy, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = validate_new(validLoader, model)
        epoch_time = time.time() - epoch_start_time
        print(f"Seed {seed} Epoch {epochind} finished, time: {epoch_time:.2f} s")

        e_tables = PrettyTable(['seed', 'epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'kappa', 'ap'])
        e_tables.float_format = '.3'
        row = [seed, epochind, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
        e_tables.add_row(row)
        print(e_tables)

        early_stop = stopper.step(ACC, model)
        if early_stop:
            best_epoch = epochind
            break

    stopper.load_checkpoint(model)
    accuracy, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = validate_new(testLoader, model)

    # 返回测试集指标（用于最终汇总）
    return {
        'seed': seed,
        'ACC': float(ACC),
        'BACC': float(BACC),
        'Prec': float(Prec),
        'Rec': float(Rec),
        'F1': float(F1),
        'AUC': float(roc_auc),
        'MCC': float(mcc),
        'kappa': float(kappa),
        'ap': float(ap),
        'best_epoch': int(best_epoch),
    }


if __name__ == "__main__":
    device = torch.device("cuda")

    seeds = [2029,2027,2026,2028,2025]  # 你想要的 5 个 seed
    all_results = []

    for sd in seeds:
        print("\n" + "=" * 80)
        print(f"Running seed = {sd}")
        print("=" * 80)
        res = run_expriments(device, sd)
        all_results.append(res)

    summarize_results(all_results)
