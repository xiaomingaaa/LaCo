import torch
import numpy as np
import time
import os
import dgl
from collections import defaultdict
from utils import eval_loader, save_model, load_model, get_parameter_number
import torch.nn.functional as F
from layers import Regularization
from model import HeteroGAT, GCN, KGNN
from dataloader import SynData, load_data
from torch.optim import Adam, SGD
from tqdm import tqdm
import argparse
import warnings
import wandb
import random

warnings.filterwarnings("ignore")

def read_triple(file_path, entity2id, relation2id):
    triples = []
    with open(file_path, 'r') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            triples.append([entity2id[h.strip()], relation2id[r.strip()], entity2id[t.strip()]])

    return triples

def construct_dgl_graph(args):
    with open(f'datasets/kg/entities.dict') as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity.strip()] = int(eid)
    with open(f'datasets/kg/relations.dict') as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation.strip()] = int(rid)
    train_triples = read_triple(f'datasets/kg/train_new.tsv', entity2id, relation2id)
    edge_dict = defaultdict(list)
    for (h, r, t) in train_triples:
        etype = ('node', str(r), 'node')
        edge_dict[etype] += [(h, t)]
    # graph = dgl.graph(num_nodes=len(entity2id))
    graph = dgl.heterograph(edge_dict, num_nodes_dict={'node':len(entity2id)})
    if args.model in ['GCN', 'GAT', 'SAGE', 'KGNN']:
        graph = dgl.to_homogeneous(graph)
        graph = dgl.add_reverse_edges(graph)

    # graph = dgl.add_reverse_edges(graph).to(self.args.device)
    graph = graph.to(args.device)
    
    return  graph, entity2id, relation2id

def build_model(graph, args):
    if args.llm in ['gpt-4o-mini', 'gpt-3.5-turbo', 'llama3-8b-chat', 'Baichuan2-chat', 'base']:
        # llm_embed_size = 768
        llm_embed_size = 1024
    elif args.llm in ['gpt-5', 'claude-2', 'llama', 'qwen']:
        llm_embed_size = 3072  
        # llm_embed_size = 1024     
    else:
        raise Exception('Unsupported LLM!')
    if args.model == 'GCN':
        print('GCN loading...')
        model = GCN(num_nodes=graph.num_nodes(), in_size=args.in_size, hid_size=args.hid_size, out_size=args.out_size, llm_embed_size=llm_embed_size)
    elif args.model == 'HGAT':
        print('HGAT loading...')
        model = HeteroGAT(etypes=graph.etypes, num_nodes=graph.num_nodes(), in_size=args.in_size, hid_size=args.hid_size, out_size=args.out_size, llm_embed_size=llm_embed_size)
    elif args.model == 'KGNN':
        print('KGNN loading...')
        model = KGNN(num_nodes=graph.num_nodes(), in_size=args.in_size, hid_size=args.hid_size, out_size=args.out_size, llm_embed_size=llm_embed_size)
    else:
        raise Exception('Unsupported Model!')

    return model

def main(args):
    wandb.init(
        # set the wandb project where this run will be logged
        project="LLM4DB",

        # track hyperparameters and run metadata
        config={
        "learning_rate": args.lr,
        "architecture": args.model,
        "dataset": args.dataset,
        "llm": args.llm,
        'setting': args.setting,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        }, mode=args.wandb_mode
    )
    # model_setting = model_config(KG=args.KG)
    train_data, valid_data, test_data = load_data(args)
    graph, entity2id, relation2id = construct_dgl_graph(args)

    model = build_model(graph, args)
    reg = Regularization(model, args.weight_decay)

    model.to(args.device)
    reg.to(args.device)

    optim = Adam(model.parameters(), lr=args.lr)
    print('Number of Parameters: ', get_parameter_number(model))
    loss_func = F.binary_cross_entropy
    early_stop = 0
    best_roc = 0.
    bar = tqdm(range(500))
    for i in bar:
        model.train()
        loss_t = 0
        start = time.time()
        for b_idx, batch in enumerate(train_data):
            h_s, t_s, c_s, d1_embed, d2_embed, c_embed, labels = batch
            h_s, t_s, c_s, d1_embed, d2_embed, c_embed = h_s.to(args.device), t_s.to(args.device), c_s.to(args.device), d1_embed.to(args.device), d2_embed.to(args.device), c_embed.to(args.device)
            labels = labels.to(args.device)
            
            preds, c_loss = model(graph, h_s, t_s, c_s, d1_embed, d2_embed, c_embed, aug=args.aug)

            optim.zero_grad()
            loss = loss_func(preds, labels.unsqueeze(-1).float())
            reg_loss = reg(model)
            if c_loss is None :
                loss_total = loss + 1*reg_loss 
            else:
                loss_total = loss + 1*reg_loss + 1*c_loss
            loss_total.backward()
            optim.step()
            loss_t+=loss_total.detach().cpu().item()
            bar.set_description(f'Training: epoch-{i+1}, {str(b_idx+1)}, loss_train: {loss_total.cpu().detach().numpy()}, c_loss: {c_loss}')
            wandb.log({'train/loss_total': loss_total.cpu().detach().numpy(), 
            'train/c_loss':c_loss, 
            'train/loss_cls':loss.cpu().detach().numpy(),
            'train/loss_reg':reg_loss.cpu().detach().numpy()
            })

        loss_print = loss_t / b_idx
        early_stop+=1
        metrics = eval_loader(model, valid_data, args.device, mc=False, graph=graph, aug=args.aug)
        (roc_auc, recall, precision, acc, aupr) = metrics['auc'], metrics['recall'], metrics['precision'], metrics['acc'], metrics['aupr']
        print('Epoch: {} | train_loss: {}, auc: {}, aupr: {}, recall: {}, precision: {}, acc: {}'.format(i+1, loss_print, roc_auc, aupr, recall, precision, acc))
        wandb.log({'train_loss': loss_print, 'val/auc': roc_auc, 'val/aupr': aupr, 'val/recall': recall, 'val/precision': precision, 'val/acc': acc, 'epoch': i+1})
        if roc_auc > best_roc:
            best_roc = roc_auc
            save_model(args, model)
            print('best model saved!!!')
            early_stop = 0
        if early_stop > 20:
            break
        end= time.time()
        print('Average Training time of one sample: {}s'.format((end-start)/len(train_data.dataset)))
    model = load_model(args, model)
    metrics = eval_loader(model, test_data, args.device, graph=graph, aug=args.aug)
    (roc_auc, recall, precision, acc, aupr) = metrics['auc'], metrics['recall'], metrics['precision'], metrics['acc'], metrics['aupr']
    print('Test_loss: auc: {}, aupr: {}, recall: {}, precision: {}, acc: {}'.format(roc_auc, aupr, recall, precision, acc))
    # wandb.log({'t'})
    wandb.finish()
    return (roc_auc, aupr, recall, precision, acc)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate of pretrain')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')

    ### classifier
    parser.add_argument('--in_size', type=int, default=128, 
                        help='The input feature size of graph learner')
    parser.add_argument('--hid_size', type=int, default=64, 
                        help='The input feature size of graph learner')
    parser.add_argument('--out_size', type=int, default=64, 
                        help='The input feature size of graph learner')
    parser.add_argument('--model', type=str, default='GCN', 
                        help='[HGAT, GCN, KGNN]')
    
    # model save
    parser.add_argument('--ckpt_dir', type=str, default='ckpts/model',
                        help='the saved root path of subgraph')
    parser.add_argument('--dataset', type=str, default='drugcomb',
                        help='the saved root path of subgraph')
    
    parser.add_argument('--flag', type=str, default='test',
                        help='the flag for current experiment, which will be used in model saving and logging')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='')
    parser.add_argument('--llm', type=str, default='gpt-4o-mini',
                        help='[gpt-4o-mini, gpt-3.5-turbo, gpt-5, llama3.1-8B, qwen2.5-7B]')
    parser.add_argument('--setting', type=str, default='S1',
                        help='long-tail setting: [S0, S1]')
    parser.add_argument('--debug', action="store_true",
                        help='debug mode with wandb dryrun')
    parser.add_argument('--aug', action="store_true",
                        help='data augmentation with llm generated descriptions')

    args = parser.parse_args()
    print(args)
    

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    if args.debug:
        args.wandb_mode = 'dryrun'
    else:
        args.wandb_mode = 'online'
    
    repeats = 5
    os.makedirs('logs', exist_ok=True)
    log = open(f'logs/{args.dataset}_{args.model}_{args.llm}_{args.setting}_{args.aug}_{args.flag}_log_llm.txt', 'w')
    results = []
    for _ in range(repeats):
        seed = random.randint(0, 10000)
        # seed = 8705
        print(f'Seed: {seed}')
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        (roc_auc, aupr, recall, precision, acc) = main(args)
        results.append([roc_auc, aupr, recall, precision, acc])
        log.write(f'Seed: {seed}, AUC: {roc_auc}, AUPR: {aupr}, Recall: {recall}, Precision: {precision}, Acc: {acc}\n')
    
    results = np.array(results)
    log.write(f'Average AUC: {results[:,0].mean()} ± {results[:,0].std()}\n')
    log.write(f'Average AUPR: {results[:,1].mean()} ± {results[:,1].std()}\n')
    log.write(f'Average Recall: {results[:,2].mean()} ± {results[:,2].std()}\n')
    log.write(f'Average Precision: {results[:,3].mean()} ± {results[:,3].std()}\n')
    log.write(f'Average Acc: {results[:,4].mean()} ± {results[:,4].std()}\n')
    log.close()