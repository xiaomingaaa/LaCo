import os
import os.path as osp
import random
import numpy as np
import torch
from mmcv.utils import collect_env as collect_base_env
from torch_geometric.loader import DataLoader
from dataset.My_inMemory_dataset import MyInMemoryDataset
from metrics import get_metrics
import torch.nn.functional as F

def kl_loss(q, p):
        kl_pq = F.kl_div(torch.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        kl_qp = F.kl_div(torch.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        sym_kl_div = kl_pq + kl_qp
        # print(sym_kl_div)
        return sym_kl_div

def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EarlyStopping():
    def __init__(self, mode='higher', patience=50, filename=None, metric=None, n_fold=None, folder=None):
        """
        Initialize EarlyStopping object.

        Args:
            mode (str): 'higher' if a higher score is better, 'lower' if a lower score is better.
            patience (int): Number of epochs to wait for improvement before early stopping.
            filename (str): Name of the checkpoint file to save the model state.
            metric (str): Metric to monitor for early stopping. Can be 'r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', or 'mse'.
            n_fold (int): Fold number used for naming checkpoint file.
            folder (str): Folder path to save checkpoint file.
        """

        if filename is None:
            filename = os.path.join(folder, '{}_fold_early_stop.pth'.format(n_fold))

        if metric is not None:
            assert metric in ['accuracy', 'f1', 'mcc', 'roc_auc', 'precision', 'recall'], \
                "Expect metric to be 'accuracy', 'f1', 'mcc', 'roc_auc', 'precision', 'recall', got {}".format(metric)
            if metric in ['accuracy', 'f1', 'mcc', 'roc_auc', 'precision', 'recall']:
                print(f'For metric {metric}, the higher the better')
                mode = 'higher'  # For all these metrics, higher is better

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False
        self.metric = metric

    def _check_higher(self, score, prev_best_score):
        """
        Check if the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """
        Check if the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):

        model.load_state_dict(torch.load(self.filename))


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    return env_info


def load_dataloader(n_fold,args):
    work_dir = args.workdir
    data_root = osp.join(work_dir,'MyDataset')

    if args.celldataset == 1:
        celllines_data = osp.join(data_root,'1024079_genes_norm.npy')
    elif args.celldataset == 2:
        celllines_data = osp.join(data_root,'1024079_genes_norm.npy')
    elif args.celldataset == 3:
        celllines_data = osp.join(data_root,'1024079_genes_norm.npy')

    
    
    drugs_data = osp.join(data_root,'LLM_data/5_l_drug_embeddings.npy')
    celllines_data = osp.join(data_root,'LLM_data/5_l_cell_embeddings.npy')
    Sdrugs_data = osp.join(data_root,'drugSmile_drugSubEmbed_db.npy')
    Scelllines_data = osp.join(data_root,'1024079_genes_norm.npy')

    split = getattr(args, 'dataset_split', 'v1')
    if split == 'v1':
        tr_data_items = osp.join(data_root, 'new-cell/v1-train.csv')
        val_data_items = osp.join(data_root, 'new-cell/v1-valid.csv')
        test_data_items = osp.join(data_root, 'new-cell/v1-test.csv')
    elif split == 'db':
        tr_data_items = osp.join(data_root, 'new-cell/db-train.csv')
        val_data_items = osp.join(data_root, 'new-cell/db-valid.csv')
        test_data_items = osp.join(data_root, 'new-cell/db-test.csv')
    elif split == 'longtail_v1':
        tr_data_items = osp.join(data_root, 'long-tail/drugcomb_train_top_cells.csv')
        val_data_items = osp.join(data_root, 'long-tail/drugcomb_test_tail_cells.csv')
        test_data_items = osp.join(data_root, 'long-tail/drugcomb_test_tail_cells.csv')
    elif split == 'longtail_db':
        tr_data_items = osp.join(data_root, 'long-tail/drugcombdb_train_top_cells.csv')
        val_data_items = osp.join(data_root, 'long-tail/drugcombdb_test_tail_cells.csv')
        test_data_items = osp.join(data_root, 'long-tail/drugcombdb_test_tail_cells.csv')
    else:
        raise ValueError(f'Unsupported dataset_split: {split}')
    
    

    tr_dataset = MyInMemoryDataset(data_root,tr_data_items,celllines_data,drugs_data,Sdrugs_data,Scelllines_data,args=args)
    val_dataset = MyInMemoryDataset(data_root,val_data_items,celllines_data,drugs_data,Sdrugs_data,Scelllines_data,args=args)
    test_dataset = MyInMemoryDataset(data_root,test_data_items,celllines_data,drugs_data,Sdrugs_data,Scelllines_data,args=args)

  
    tr_dataloader = DataLoader(tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)
    
    print(f'train data:{len(tr_dataloader)*args.batch_size}')  
    print(f'Valid data:{len(val_dataloader)*args.batch_size}')
    print(f'Test data:{len(test_dataloader)*args.batch_size}')
    
    return tr_dataloader, val_dataloader, test_dataloader



def load_infer_dataloader(args):
    """
    推理用 dataloader，和训练保持同一套数据根目录/embedding 路径。
    data_items 由 args.infer_path 控制，可以是 npy 或 csv。
    """

    work_dir = args.workdir
    data_root = osp.join(work_dir, 'MyDataset')

    # 如果你现在 celllines_data / Scelllines_data 实际上都用同一个 1024079_genes_norm.npy，
    # 就暂时跟 load_dataloader 保持一致
    # （注意：下面这几行和 load_dataloader 里的保持同样写法）
    if args.celldataset == 1:
        base_cell_file = '1024079_genes_norm.npy'
    elif args.celldataset == 2:
        base_cell_file = '1024079_genes_norm.npy'
    elif args.celldataset == 3:
        base_cell_file = '1024079_genes_norm.npy'
    else:
        base_cell_file = '1024079_genes_norm.npy'

    # LLM 表征 & 原始 omics
    drugs_data = osp.join(data_root, 'LLM_data/5_l_drug_embeddings.npy')
    celllines_data = osp.join(data_root, 'LLM_data/5_l_cell_embeddings.npy')  # LLM cell
    Sdrugs_data = osp.join(data_root, 'drugSmile_drugSubEmbed_db.npy')
    Scelllines_data = osp.join(data_root, base_cell_file)                    # 原始 omics

    # 推理用的样本列表（路径由 args.infer_path 决定）
    data_items = args.infer_path

    # 和训练同样的 MyInMemoryDataset 调用方式
    infer_dataset = MyInMemoryDataset(
        data_root,
        data_items,
        celllines_data,
        drugs_data,
        Sdrugs_data,
        Scelllines_data,
        args=args
    )

    # 推理时通常 batch_size 可根据需要设：
    #   - 做 UMAP 想要精确 one-by-one，可用 1
    #   - 想加速可以直接用 args.batch_size
    infer_dataloader = DataLoader(
        infer_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    # infer_data_arr：你后面要拼 output 时用到 drugA, drugB, sample_id, label
    # 之前是 np.load(data_items)，现在要根据文件类型做一点处理
    if data_items.endswith('.npy'):
        infer_data_arr = np.load(data_items, allow_pickle=True)
    else:
        # 如果你以后把 infer_path 换成 csv，可以这样处理：
        import pandas as pd
        df = pd.read_csv(data_items)
        # 根据你 csv 里的列名取出用于拼接的列
        # 这里假设和训练一样有这四列
        infer_data_arr = df[['drug_a', 'drug_b', 'cell', 'synergy']].values

    return infer_dataloader, infer_data_arr




def train(model, criterion, opt, dataloader, device, args=None):
    model.train()
    train_loss_sum = 0
    contrastive_loss_sum = 0  # 用来累计对比学习损失
    i = 0 
    lr_list = []

    for data in dataloader:
        i += 1
        model.zero_grad()
        x = data.to(device)
        output,contrastive_loss ,_, _ = model(x)  # 假设模型返回对比损失
        y = data.y.unsqueeze(1).type(torch.float32).to(device)

        # 计算总损失（包括标准损失和对比损失）
        train_loss = criterion(output, y) 
        # klloss = kl_loss(output, output2)
        train_loss=train_loss+contrastive_loss
        # print(klloss.item())
        train_loss_sum = train_loss_sum+train_loss.item()
        # + contrastive_loss.item() # 累积训练损失
        # contrastive_loss_sum += contrastive_loss.item()  # 累积对比损失

        # 反向传播和优化器更新
        train_loss.backward()
        
        opt.step()
    


    # 计算训练损失和对比损失的平均值
    avg_train_loss = train_loss_sum / i
    

    return avg_train_loss, lr_list

        





def validate(model,criterion,dataloader,device,args=None):

    model.eval()
    y_true = []
    y_pred = []
    i = 0
    
    with torch.no_grad():
        for data in dataloader:
            i += 1
            x = data.to(device) 
            y = data.y.unsqueeze(1).to(device)
            y_true.append(y.view(-1, 1))
            output, _,_, _ = model(x)
            y_pred.append(output)            

    y_true = torch.cat(y_true, dim=0).cpu().detach().numpy()
    y_pred = torch.cat(y_pred, dim=0).cpu().detach().numpy()      
    accuracy, f1, mcc, roc_auc, kappa, ap, precision, recall = get_metrics(y_true, y_pred)

    return accuracy, f1, mcc, roc_auc, kappa, ap, precision, recall 





def infer(model, dataloader, device, args=None):

    model.eval()

    y_pred_list = []
    cell_embed_list = []
    attn_list = []

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)

            # 当前 SynergyxNet.forward 返回 4 个值：
            # logits: 模型输出（未过 sigmoid）
            # contrastive_loss: 推理阶段不使用
            # cell_embed: 模型内部 cell 表征
            # attn: 选定层的 attention
            logits, cell_embed, attn = model(data)

            # 如果训练用的是 BCEWithLogitsLoss，这里需要 sigmoid
            prob = torch.sigmoid(logits)   # [B, 1] / [B]
            y_pred_list.append(prob.cpu())

            # 只有在用户要求输出 attn / embedding 时才存
            if args is not None and args.output_attn:
                cell_embed_list.append(cell_embed.cpu())
                attn_list.append(attn.cpu())

    # ===== 拼接所有 batch =====
    y_pred = torch.cat(y_pred_list, dim=0).numpy()

    if len(cell_embed_list) > 0:
        cell_embed = torch.cat(cell_embed_list, dim=0).numpy()
    else:
        cell_embed = None

    if len(attn_list) > 0:
        attn = torch.cat(attn_list, dim=0).numpy()
    else:
        attn = None

    return y_pred, cell_embed, attn
