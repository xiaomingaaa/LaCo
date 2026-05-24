import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from loss_util import sce_loss, contrastive_loss_v2, alignment
import numpy as np
from dgl import apply_each
from dgl.dataloading import DataLoader, NeighborSampler


class ResidualLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        # 定义一个全连接层
        self.fc = nn.Linear(input_dim, output_dim)
        self.adjust_dim = nn.Linear(input_dim, output_dim)
        print('Residual')
        
    def forward(self, x):
        # 全连接层的输出
        out = F.relu(self.fc(x))
        
        # 残差连接
        # 只有当输入和输出维度相同时才能直接相加
        if x.size() == out.size():
            out = out + x
        else:
            # 如果维度不同，可以通过一个额外的线性变换调整维度
            # 这里需要定义一个额外的线性层用于维度匹配（可以在 __init__ 中定义）
            # 例如: self.adjust_dim = nn.Linear(input_dim, output_dim)
            out = out + self.adjust_dim(x)
        return out

class HeteroGAT(nn.Module):
    def __init__(self, etypes, num_nodes, in_size, hid_size, out_size, n_heads=4, llm_embed_size=1024):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.embed = nn.Parameter(torch.rand(num_nodes, in_size), requires_grad=True)
        nn.init.xavier_normal_(self.embed)
        # entity_embed = torch.from_numpy(np.load('ckpts/TransE_primary_kg_0/entity_embedding.npy'))
        # self.embed = nn.Parameter(entity_embed, requires_grad=True)
        
        self.layers.append(
            dglnn.HeteroGraphConv(
                {
                    etype: dglnn.GATConv(in_size, hid_size // n_heads, n_heads)
                    for etype in etypes
                }
            )
        )
        self.dropout = nn.Dropout(0.5)
        self.output_layers = nn.ModuleList()
        self.output_layers.append(
            nn.Linear(hid_size, out_size)
        )
        # self.output_layers.append(
        #     nn.Linear(out_size, 1)
        # )
        self.adapter = nn.Linear(llm_embed_size, out_size)
        self.linear = nn.Linear(out_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.residual = ResidualLayer(out_size*3, out_size)
        # self.loss_func = contrastive_loss_v2
        self.loss_func = self.alignment
        
    def _mask_features(self, features, mask_ratio=0.2):
        """Randomly mask a portion of the features."""
        num_nodes, feat_dim = features.size()
        mask = torch.rand(num_nodes, feat_dim, device=features.device) < mask_ratio
        masked_features = features.clone()
        masked_features[mask] = 0
        return masked_features
    
    def alignment(self, x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()
    
    def forward(self, g, h_id, t_id, c_id, d1_embed, d2_embed, cell_embed, aug=True, mask_ratio=0.2):
        h = {'node':self.embed}
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            # One thing is that h might return tensors with zero rows if the number of dst nodes
            # of one node type is 0.  x.view(x.shape[0], -1) wouldn't work in this case.
            h = apply_each(
                h, lambda x: x.view(x.shape[0], x.shape[1] * x.shape[2])
            )
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)

        g_embed = h['node']

        for layer in self.output_layers:
            g_embed = layer(g_embed)
        
        h_embed = g_embed[h_id]
        t_embed = g_embed[t_id]
        c_embed = g_embed[c_id]
        # from ipdb import set_trace;set_trace()
        if aug:
            loss1 = self.loss_func(h_embed.float(), self.adapter(d1_embed.float()))
            loss2 = self.loss_func(t_embed.float(), self.adapter(d2_embed.float()))
            loss3 = self.loss_func(c_embed.float(), self.adapter(cell_embed.float()))

        embed = torch.cat([
            h_embed, t_embed, c_embed
        ], dim=1)
        embed = self.residual(embed)
        logits = self.linear(embed)
        pred = self.sigmoid(logits)
        if aug:
            return pred, loss1+loss2+loss3
        else:
            return pred, None

class GCN(nn.Module):
    def __init__(self, num_nodes, in_size, hid_size, out_size, llm_embed_size=768):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.embed = nn.Parameter(torch.rand(num_nodes, in_size), requires_grad=True)
        nn.init.xavier_normal_(self.embed)
        # entity_embed = torch.from_numpy(np.load('ckpts/TransE_primary_kg_0/entity_embedding.npy'))
        # self.embed = nn.Parameter(entity_embed, requires_grad=True)
        
        self.layers.extend([
            dglnn.GraphConv(in_size, hid_size, allow_zero_in_degree=True),
            # dglnn.GraphConv(hid_size, hid_size, allow_zero_in_degree=True),
        ]
        )
        
        self.dropout = nn.Dropout(0.5)
        self.output_layers = nn.ModuleList()
        self.output_layers.append(
            nn.Linear(hid_size, out_size)
        )

        self.adapter = nn.Linear(llm_embed_size, hid_size)
        self.linear = nn.Linear(out_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.residual = ResidualLayer(out_size*3, out_size)
        self.loss_func = contrastive_loss_v2
        # self.loss_func = self.alignment
        
    def contrastive_loss_v2(self, feat, llm_feat, temperature=0.07):
        feat = F.normalize(feat, p=2, dim=1)
        llm_feat = F.normalize(llm_feat, p=2, dim=1)
        similarity_matrix = torch.matmul(feat, llm_feat.T)

        mask = torch.eye(feat.size(0), device=feat.device).bool()
        positive_similarity = similarity_matrix[mask].view(feat.size(0), -1)
        negative_similarity = similarity_matrix[~mask].view(feat.size(0), -1)

        logits = torch.cat([positive_similarity, negative_similarity], dim=1)
        logits /= temperature
        labels = torch.zeros(feat.size(0), dtype=torch.long).to(feat.device)
        loss = F.cross_entropy(logits, labels)

        return loss

    def alignment(self, x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()

    def contrastive_loss(self, feat, llm_feat, temperature=0.07):
        batch_size = feat.shape[0]

        similarity_matrix = F.cosine_similarity(feat.unsqueeze(1), llm_feat.unsqueeze(0), dim=2) / temperature

        labels = torch.eye(batch_size, device=feat.device).view(1, -1)
        logits = F.sigmoid(similarity_matrix).view(1, -1)
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, g, h_id, t_id, c_id, d1_embed, d2_embed, cell_embed, aug=True):
        h = self.embed
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)

        for layer in self.output_layers:
            h = layer(h)
        
        h_embed = h[h_id]
        t_embed = h[t_id]
        c_embed = h[c_id]
        # from ipdb import set_trace;set_trace()
        if aug:
            loss1 = self.loss_func(h_embed, self.adapter(d1_embed))
            loss2 = self.loss_func(t_embed, self.adapter(d2_embed))
            loss3 = self.loss_func(c_embed, self.adapter(cell_embed))
        
        embed = torch.cat([
            h_embed, t_embed, c_embed
        ], dim=1)
        embed = self.residual(embed)
        logits = self.linear(embed)
        pred = self.sigmoid(logits)
        if aug:
            return pred, loss1+loss2+loss3
        else:
            return pred, None

class KGNN(nn.Module):
    def __init__(self, num_nodes, in_size, hid_size, out_size, llm_embed_size=768):
        super().__init__()
        self.layers = nn.ModuleList()
        
        self.embed = nn.Parameter(torch.rand(num_nodes, in_size), requires_grad=True)
        nn.init.xavier_normal_(self.embed)
        # entity_embed = torch.from_numpy(np.load('ckpts/TransE_primary_kg_0/entity_embedding.npy'))
        # self.embed = nn.Parameter(entity_embed, requires_grad=True)
        
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        
        self.dropout = nn.Dropout(0.5)
        self.output_layers = nn.ModuleList()
        self.output_layers.append(
            nn.Linear(hid_size, out_size)
        )

        self.adapter = nn.Linear(llm_embed_size, hid_size)
        self.residual = ResidualLayer(out_size*3, out_size)
        self.linear = nn.Linear(out_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_func = contrastive_loss_v2
        # self.loss_func = self.alignment
        

    def contrastive_loss_v2(self, feat, llm_feat, temperature=0.07):
        feat = F.normalize(feat, p=2, dim=1)
        llm_feat = F.normalize(llm_feat, p=2, dim=1)
        similarity_matrix = torch.matmul(feat, llm_feat.T)

        mask = torch.eye(feat.size(0), device=feat.device).bool()
        positive_similarity = similarity_matrix[mask].view(feat.size(0), -1)
        negative_similarity = similarity_matrix[~mask].view(feat.size(0), -1)

        logits = torch.cat([positive_similarity, negative_similarity], dim=1)
        logits /= temperature
        labels = torch.zeros(feat.size(0), dtype=torch.long).to(feat.device)
        loss = F.cross_entropy(logits, labels)

        return loss
    
    def alignment(self, x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean()   

    def contrastive_loss(self, feat, llm_feat, temperature=0.07):
        batch_size = feat.shape[0]

        similarity_matrix = F.cosine_similarity(feat.unsqueeze(1), llm_feat.unsqueeze(0), dim=2) / temperature

        labels = torch.eye(batch_size, device=feat.device).view(1, -1)

        logits = F.sigmoid(similarity_matrix).view(1, -1)

        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, g, h_id, t_id, c_id, d1_embed, d2_embed, cell_embed, aug=False):
        h = self.embed
        for l, layer in enumerate(self.layers):
            h = layer(g, h)
            
            if l != len(self.layers) - 1:
                h = apply_each(h, F.relu)
                h = apply_each(h, self.dropout)

        for layer in self.output_layers:
            h = layer(h)
        
        h_embed = h[h_id]
        t_embed = h[t_id]
        c_embed = h[c_id]
    
        if aug:
            d1_embed = self.adapter(d1_embed.float())
            d2_embed = self.adapter(d2_embed.float())
            cell_embed = self.adapter(cell_embed.float())
            loss1 = self.loss_func(h_embed, d1_embed)
            loss2 = self.loss_func(t_embed, d2_embed)
            loss3 = self.loss_func(c_embed, cell_embed)
            h_embed = h_embed + d1_embed
            t_embed = t_embed + d2_embed
            c_embed = c_embed + cell_embed
        
        embed = torch.cat([
            h_embed, t_embed, c_embed
        ], dim=1)

        embed = self.residual(embed)
        logits = self.linear(embed)
        pred = self.sigmoid(logits)
        if aug:
            return pred, loss1+loss2+loss3
        else:
            return pred, None