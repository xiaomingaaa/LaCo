import random
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch.utils.data as Data
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import TensorDataset, Dataset
from models.gat import GATNet
from models.gat_gcn_test import GAT_GCN
from models.gcn import GCNNet
from models.ginconv import GINConvNet
from utils_test import *
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score, roc_auc_score, precision_score, recall_score, balanced_accuracy_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, cohen_kappa_score, average_precision_score, precision_score, recall_score, confusion_matrix
import copy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def kl_loss(q, p):
    kl_pq = F.kl_div(torch.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    kl_qp = F.kl_div(torch.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    sym_kl_div = kl_pq + kl_qp
    return sym_kl_div

def alignment( x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

# training function at each epoch
def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch):
    print('Training on {} samples...'.format(len(drug1_loader_train.dataset)))
    model.train()
    # train_loader = np.array(train_loader)
    for batch_idx, data in enumerate(zip(drug1_loader_train, drug2_loader_train)):
        data1 = data[0]
        data2 = data[1]
        data1 = data1.to(device)
        data2 = data2.to(device)
        y = data[0].y.view(-1, 1).long().to(device)
        y = y.squeeze(1)
        optimizer.zero_grad()
        output,output_llm,contrastive_loss = model(data1, data2)
        # kl_loss_llm = alignment(output, output_llm)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(output, y)
        # loss = loss+kl_loss_llm
        loss = loss+contrastive_loss
        # print('loss', loss)
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data1.x),
                                                                           len(drug1_loader_train.dataset),
                                                                           100. * batch_idx / len(drug1_loader_train),
                                                                           loss.item()))


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(drug1_loader_test.dataset)))
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0]
            data2 = data[1]
            data1 = data1.to(device)
            data2 = data2.to(device)
            output,output_llm,contrastive_loss = model(data1, data2)
            ys = F.softmax(output, 1).to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: np.argmax(x), ys))
            predicted_scores = list(map(lambda x: x[1], ys))
            total_preds = torch.cat((total_preds, torch.Tensor(predicted_scores)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

# modeling = [GINConvNet, GATNet, GAT_GCN, GCNNet][int(sys.argv[2])]
# datasets = [['davis', 'kiba'][int(sys.argv[1])]]
# model_st = modeling.__name__
modeling = GCNNet

TRAIN_BATCH_SIZE = 256
TEST_BATCH_SIZE = 256
LR = 0.0005
LOG_INTERVAL = 20
NUM_EPOCHS = 2000

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
datafile = 'drugcomb/filtered_updated_drugcomb_v1'

# CPU or GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU...')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

lenth = len(drug1_data)
pot = int(lenth/5)
print('lenth', lenth)
print('pot', pot)


random_num = random.sample(range(0, lenth), lenth)
best_auc_per_fold = []

SEEDS = random.sample(range(1, 10**6), 5)
print("Random seeds for this run:", SEEDS)
metrics_per_run = {
    'AUC': [],
    'PR_AUC': [],
    'ACC': [],
    'F1': [],
    'MCC': [],
    'Kappa': [],
    'Precision': [],
    'Recall': [],
    'TPR': []
}

for seed in SEEDS:
    print(f"\n=== Running experiment with seed {seed} ===")
    # Set random seed
    set_seed(seed)

    # Random shuffle indices for splitting
    random_num = list(range(len(drug1_data)))  # 假设两者长度相同
    random.shuffle(random_num)
    # split_point = int(0.8 * len(random_num))
    # train_num = random_num[:split_point]
    # test_num = random_num[split_point:]
    
    train_idx = np.loadtxt('data/drugcombdb/inductive/train_idx.txt', dtype=int)
    valid_idx = np.loadtxt('data/drugcombdb/inductive/valid_idx.txt', dtype=int)
    test_idx  = np.loadtxt('data/drugcombdb/inductive/test_idx.txt', dtype=int)

    drug1_data = TestbedDataset(root='data', dataset=datafile + '_drug1')
    drug2_data = TestbedDataset(root='data', dataset=datafile + '_drug2')

    drug1_train = drug1_data[train_idx]
    drug1_valid = drug1_data[valid_idx]
    drug1_test  = drug1_data[test_idx]

    drug2_train = drug2_data[train_idx]
    drug2_valid = drug2_data[valid_idx]
    drug2_test  = drug2_data[test_idx]

    
    drug1_loader_train = DataLoader(drug1_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    drug1_loader_valid = DataLoader(drug1_valid, batch_size=TEST_BATCH_SIZE, shuffle=False)
    drug1_loader_test  = DataLoader(drug1_test, batch_size=TEST_BATCH_SIZE, shuffle=False)

    drug2_loader_train = DataLoader(drug2_train, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    drug2_loader_valid = DataLoader(drug2_valid, batch_size=TEST_BATCH_SIZE, shuffle=False)
    drug2_loader_test  = DataLoader(drug2_test, batch_size=TEST_BATCH_SIZE, shuffle=False)

    
    
    
    

    # Prepare loaders
#     drug1_loader_train = DataLoader(drug1_data[train_num], batch_size=TRAIN_BATCH_SIZE, shuffle=True)
#     drug1_loader_test = DataLoader(drug1_data[test_num], batch_size=TRAIN_BATCH_SIZE, shuffle=False)

#     drug2_loader_train = DataLoader(drug2_data[train_num], batch_size=TRAIN_BATCH_SIZE, shuffle=True)
#     drug2_loader_test = DataLoader(drug2_data[test_num], batch_size=TRAIN_BATCH_SIZE, shuffle=False)

    # Init model
    model = modeling().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    
    best_val_auc = 0.0
    best_model_state = None
    patience = 100
    early_stop_counter = 0
    
    # Train
    for epoch in range(NUM_EPOCHS):
        train(model, device, drug1_loader_train, drug2_loader_train, optimizer, epoch + 1)
        T_val, S_val, Y_val = predicting(
            model, device, drug1_loader_valid, drug2_loader_valid
        )
        val_auc = roc_auc_score(T_val, S_val)

        print(f"[Seed {seed}] Epoch {epoch+1} | Val AUC: {val_auc:.4f}")

        # —— 保存最优模型 ——
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = copy.deepcopy(model.state_dict())
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        # —— Early Stopping ——
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


    # Evaluate
    model.load_state_dict(best_model_state)

    T, S, Y = predicting(
        model, device, drug1_loader_test, drug2_loader_test
    )

    # T, S, Y = predicting(model, device, drug1_loader_valid, drug2_loader_valid)

    AUC = roc_auc_score(T, S)
    PR_AUC = average_precision_score(T, S)
    ACC = accuracy_score(T, Y)
    F1 = f1_score(T, Y)
    MCC = matthews_corrcoef(T, Y)
    Kappa = cohen_kappa_score(T, Y)
    Precision = precision_score(T, Y)
    Recall = recall_score(T, Y)
    tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
    TPR = tp / (tp + fn)

    print(f"Seed {seed} Metrics:")
    print(f"AUC: {AUC:.4f}, PR_AUC: {PR_AUC:.4f}, ACC: {ACC:.4f}, F1: {F1:.4f}, MCC: {MCC:.4f}")
    print(f"Kappa: {Kappa:.4f}, Precision: {Precision:.4f}, Recall: {Recall:.4f}, TPR: {TPR:.4f}")

    # Save this run
    metrics_per_run['AUC'].append(AUC)
    metrics_per_run['PR_AUC'].append(PR_AUC)
    metrics_per_run['ACC'].append(ACC)
    metrics_per_run['F1'].append(F1)
    metrics_per_run['MCC'].append(MCC)
    metrics_per_run['Kappa'].append(Kappa)
    metrics_per_run['Precision'].append(Precision)
    metrics_per_run['Recall'].append(Recall)
    metrics_per_run['TPR'].append(TPR)

    
# === Per-seed results ===
print("\n=== Per-seed Metrics Summary ===")
for i, seed in enumerate(SEEDS):
    print(f"Seed {seed} Metrics:")
    for metric in metrics_per_run:
        print(f"  {metric}: {metrics_per_run[metric][i]:.4f}")
    print("-" * 40)

# === After all seeds
print("\n=== Final Results over 5 Seeds ===")
for metric, values in metrics_per_run.items():
    mean = np.mean(values)
    std = np.std(values)
    print(f"{metric}: {mean:.4f} ± {std:.4f}")
    
