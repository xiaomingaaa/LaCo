from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.metrics import f1_score, recall_score, precision_score
import utils
from prettytable import PrettyTable
from model import HGANDDS
import torch.nn.functional as F
from dgllife.utils import EarlyStopping
from transformers import AdamW
from dataset import DrugSynergyDataset

import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import random
import os
import argparse
import pandas as pd

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
def contrastive_loss_v2(feat, llm_feat, temperature=1, hard_negative=False):
    """
    基于余弦相似度的对比损失函数，支持硬负样本采样。
    
    feat: 从模型输出的真实嵌入（如 drug_embed 或 cell_embed）
    llm_feat: 从大模型输出的嵌入（如 LLM_drugA 或 LLM_cell）
    temperature: 温度系数，用于调整嵌入相似度的尺度
    hard_negative: 是否启用硬负样本采样
    """
    feat = F.normalize(feat, p=2, dim=1)  # 对特征进行L2归一化
    llm_feat = F.normalize(llm_feat, p=2, dim=1)  # 对LLM特征进行L2归一化
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(feat, llm_feat.T)  # [batch_size, batch_size]
    
    # 构建正样本和负样本对
    mask = torch.eye(feat.size(0), device=feat.device).bool()  # 选择正样本的对角线
    positive_similarity = similarity_matrix[mask].view(feat.size(0), -1)
    
    if hard_negative:
        # 选择与正样本相似度较高的负样本（硬负样本）
        negative_similarity, _ = torch.topk(similarity_matrix, k=2, dim=1, largest=True)
        negative_similarity = negative_similarity[:, 1]  # 排除自身
    else:
        negative_similarity = similarity_matrix[~mask].view(feat.size(0), -1)
    
    # 合并正负样本对的相似度
    logits = torch.cat([positive_similarity, negative_similarity], dim=1)
    logits /= temperature  # 温度缩放

    # 构建标签，正样本标签为0（因为我们希望最大化正样本的概率）
    labels = torch.zeros(feat.size(0), dtype=torch.long).to(feat.device)
    
    # 使用交叉熵计算损失
    loss = F.cross_entropy(logits, labels)
    
    return loss


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()
    return (p_loss + q_loss) / 2


def read_data_file(data_file, args):
    """
    推荐你的 CSV 列为：drug_1, drug_2, context, label
    如果列名不同，用 args 指定（--col_drug1 等）。
    """
    df = pd.read_csv(data_file)

    drug1_col = args.col_drug1
    drug2_col = args.col_drug2
    context_col = args.col_context
    label_col = args.col_label

    for c in [drug1_col, drug2_col, context_col, label_col]:
        if c not in df.columns:
            raise ValueError(f"{data_file} 缺少列: {c}，当前列：{list(df.columns)}")

    drug_1 = df[drug1_col].astype(str).tolist()
    drug_2 = df[drug2_col].astype(str).tolist()
    cell_line = df[context_col].astype(str).tolist()
    Y = df[label_col].astype(int).tolist()

    return drug_1, drug_2, cell_line, Y


def define_dataloader_fixed(train_file, valid_file, test_file,
                            maxCompoundLen, batch_size, device, dataset_type, args):
    train_drug_1, train_drug_2, train_cell_line, train_Y = read_data_file(train_file, args)
    valid_drug_1, valid_drug_2, valid_cell_line, valid_Y = read_data_file(valid_file, args)
    test_drug_1, test_drug_2, test_cell_line, test_Y = read_data_file(test_file, args)

    train_dataset = DrugSynergyDataset(
        np.array(train_drug_1), np.array(train_drug_2), np.array(train_Y),
        np.array(train_cell_line), device, maxCompoundLen, dataset_type=dataset_type
    )
    valid_dataset = DrugSynergyDataset(
        np.array(valid_drug_1), np.array(valid_drug_2), np.array(valid_Y),
        np.array(valid_cell_line), device, maxCompoundLen, dataset_type=dataset_type
    )
    test_dataset = DrugSynergyDataset(
        np.array(test_drug_1), np.array(test_drug_2), np.array(test_Y),
        np.array(test_cell_line), device, maxCompoundLen, dataset_type=dataset_type
    )

    # 注意：shuffle=True 只是打乱训练 batch 顺序，不会改变 train/valid/test 划分
    trainLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validLoader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    testLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return trainLoader, validLoader, testLoader


def validate_new(valid_loader, model, comb_data, device):
    model.eval()
    preds = torch.Tensor()
    trues = torch.Tensor()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            (compounds_1, compounds_2, synergyScores, cell_line, fp1, fp2,
 cid_nodeid_list1, cid_nodeid_list2, cell_node_list,
 drug1_embed, drug2_embed, cell_embed) = batch

            compounds_1 = compounds_1.to(device)
            compounds_2 = compounds_2.to(device)
            synergyScores = synergyScores.to(device)
            cell_line = cell_line.to(device)
            fp1 = fp1.to(device)
            fp2 = fp2.to(device)
            cid_nodeid_list1 = cid_nodeid_list1.to(device)
            cid_nodeid_list2 = cid_nodeid_list2.to(device)
            cell_node_list = cell_node_list.to(device)
            drug1_embed = drug1_embed.to(device)
            drug2_embed = drug2_embed.to(device)
            cell_embed  = cell_embed.to(device)


            pre_synergy, _ = model(
            comb_data,
            cid_nodeid_list1, cid_nodeid_list2, cell_node_list,
            drug1_embed, drug2_embed, cell_embed
             )
            pre_synergy = torch.nn.functional.softmax(pre_synergy, dim=1)[:, 1]
            preds = torch.cat((preds, pre_synergy.cpu()), 0)
            trues = torch.cat((trues, synergyScores.view(-1, 1).cpu()), 0)

        y_pred = np.array(preds) > 0.5
        roc_auc = roc_auc_score(trues, preds)
        acc = accuracy_score(trues, y_pred)
        f1 = f1_score(trues, y_pred, average='binary')
        prec = precision_score(trues, y_pred, average='binary')
        rec = recall_score(trues, y_pred, average='binary')
        aupr = average_precision_score(trues, preds)
        return acc, prec, rec, f1, roc_auc, aupr


def train_one_epoch(train_loader, model, epoch, optimizer, device, scheduler, comb_data, print_freq=50):
    model.train()
    cross_entropy_loss = nn.CrossEntropyLoss()
    losses = AverageMeter()
    

    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()

        compounds_1, compounds_2, synergyScores, cell_line, fp1, fp2, cid_nodeid_list1, cid_nodeid_list2, cell_node_list, drug1_embed, drug2_embed, cell_embed = batch

        compounds_1 = compounds_1.to(device)
        compounds_2 = compounds_2.to(device)
        synergyScores = synergyScores.to(device)
        cell_line = cell_line.to(device)
        fp1 = fp1.to(device)
        fp2 = fp2.to(device)
        drug1_embed = drug1_embed.to(device)
        drug2_embed = drug2_embed.to(device)
        cell_embed  = cell_embed.to(device)

        cid_nodeid_list1 = cid_nodeid_list1.to(device)
        cid_nodeid_list2 = cid_nodeid_list2.to(device)
        cell_node_list = cell_node_list.to(device)

        pre_synergy,contrastive_loss = model(
            comb_data,
            cid_nodeid_list1, cid_nodeid_list2, cell_node_list,
            drug1_embed, drug2_embed, cell_embed
        )

        pre_synergy2, contrastive_loss  = model(
            comb_data,
            cid_nodeid_list1, cid_nodeid_list2, cell_node_list,
            drug1_embed, drug2_embed, cell_embed
        )

        ce_loss = 0.5 * (cross_entropy_loss(pre_synergy, synergyScores.squeeze(1)) +
                         cross_entropy_loss(pre_synergy2, synergyScores.squeeze(1)))
        kl_loss = compute_kl_loss(pre_synergy, pre_synergy2)
        alpha = 5
     
        beta = 0.5 
        loss = ce_loss + alpha * kl_loss+beta * contrastive_loss


        losses.update(loss.item(), len(compounds_1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % print_freq == 0:
            print('TRAIN -> Epoch{epoch}: \tIter:{iter}\t Loss:{loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch=epoch, iter=i, loss=losses
            ))


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


class AverageMeter(object):
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


def run_fixed_split(device, comb_data, train_file, valid_file, test_file, log_filename, args):
    seed_torch(args.seed)

    trainLoader, validLoader, testLoader = define_dataloader_fixed(
        train_file=train_file,
        valid_file=valid_file,
        test_file=test_file,
        maxCompoundLen=args.maxCompoundLen,
        batch_size=args.batch_size,
        device=device,
        dataset_type=args.data_type,
        args=args
    )

    model = HGANDDS(
        data=comb_data,
        hidden_channels=args.hidden_channels,
        is_gnn=False,
        drug_feature_length=args.drug_feature_length,
        data_type=args.data_type
    )

    filename = utils.add_time_suffix('hgandds_model_' + args.data_type)
    stopper = EarlyStopping(mode='higher', filename=filename, patience=args.patience)

    model = model.to(device)
    comb_data = comb_data.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, epochs=args.n_epochs, steps_per_epoch=len(trainLoader)
    )

    for epochind in range(args.n_epochs):
        train_one_epoch(trainLoader, model, epochind, optimizer, device, scheduler, comb_data, print_freq=args.print_freq)
        acc, prec, rec, f1, roc_auc, aupr = validate_new(validLoader, model, comb_data, device)

        e_tables = PrettyTable(['epoch', 'acc', 'pre', 'rec', 'f1', 'auc', 'aupr'])
        e_tables.float_format = '.3'
        e_tables.add_row([epochind, acc, prec, rec, f1, roc_auc, aupr])
        utils.log_to_file_and_console(log_file_name=log_filename, fmt='', log=e_tables)

        if stopper.step(acc, model):
            break

    stopper.load_checkpoint(model)

    acc, prec, rec, f1, roc_auc, aupr = validate_new(testLoader, model, comb_data, device)
    e_tables = PrettyTable(['test', 'acc', 'pre', 'rec', 'f1', 'auc', 'aupr'])
    e_tables.float_format = '.3'
    e_tables.add_row([epochind, acc, prec, rec, f1, roc_auc, aupr])
    utils.log_to_file_and_console(log_file_name=log_filename, fmt='', log=e_tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HGANDDS training (fixed train/valid/test split)')

    parser.add_argument('--data_type', type=str, default='drugcombdb', help="Types of drug combination data sets")
    parser.add_argument('--gpu_index', type=int, default=0, help="GPU index for training")

    # 你已有的三份数据
    parser.add_argument('--train_file', type=str, default="data/drugcombdb/db/db-train_final.csv", help="Path to train csv")
    parser.add_argument('--valid_file', type=str, default="data/drugcombdb/db/db-valid_final.csv", help="Path to valid csv")
    parser.add_argument('--test_file',  type=str, default="data/drugcombdb/db/db-test_final.csv",  help="Path to test csv")


    # 如果你的列名不是 drug_1/drug_2/context/label，用下面参数改
    parser.add_argument('--col_drug1', type=str, default='drug_1')
    parser.add_argument('--col_drug2', type=str, default='drug_2')
    parser.add_argument('--col_context', type=str, default='context')
    parser.add_argument('--col_label', type=str, default='label')

    parser.add_argument('--hidden_channels', type=int, default=64, help="The number of hidden neurons in the model")
    parser.add_argument('--drug_feature_length', type=int, default=384, help="Dimensions of drug features")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size of training data")
    parser.add_argument('--n_epochs', type=int, default=200, help="Epochs of training")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate in training process")

    parser.add_argument('--patience', type=int, default=50, help="EarlyStopping patience")
    parser.add_argument('--maxCompoundLen', type=int, default=128, help="Max compound token length")
    parser.add_argument('--seed', type=int, default=2025, help="Random seed for training reproducibility")
    parser.add_argument('--print_freq', type=int, default=50, help="Print frequency in iterations")

    args = parser.parse_args()

    data_type = args.data_type
    gpu_index = args.gpu_index
    log_filename = utils.generate_log_filename('HGANDDS_{}'.format(data_type))

    utils.log_to_file_and_console(log_file_name=log_filename, fmt='args:{}'.format(args), log=None)

    device = torch.device("cuda", gpu_index)

    comb_data = utils.init_hetero_data(
        data_type=data_type, device=device
    )

    run_fixed_split(
        device=device,
        comb_data=comb_data,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        log_filename=log_filename,
        args=args
    )
