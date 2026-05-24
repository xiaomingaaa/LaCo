import torch
import torch.nn as nn
import torch.nn.functional as F
from .contrastive import contrastive_loss_v2
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from head import FusionHead

def random_feature_masking(x, mask_ratio=0.4):
    """
    x: tensor of shape [B, D]
    Randomly masks mask_ratio percent of dimensions by setting them to 0.
    """
    B, D = x.shape
    mask = torch.ones_like(x)
    num_mask = int(D * mask_ratio)
    for i in range(B):
        idx = torch.randperm(D)[:num_mask]
        mask[i, idx] = 0
    return x * mask


# GCN based model
class GCNNet(torch.nn.Module):
    def __init__(self, n_output=2, n_filters=32, embed_dim=128,num_features_xd=78, num_features_xt=954, output_dim=768, dropout=0.2):

        super(GCNNet, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # SMILES1 graph branch
        self.n_output = n_output
        self.drug1_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug1_conv2 = GCNConv(num_features_xd, num_features_xd*2)
        self.drug1_conv3 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.drug1_fc_g1 = torch.nn.Linear(num_features_xd*4, num_features_xd*2)
        self.drug1_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)

        # SMILES2 graph branch
        self.drug2_conv1 = GCNConv(num_features_xd, num_features_xd)
        self.drug2_conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.drug2_conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.drug2_fc_g1 = torch.nn.Linear(num_features_xd * 4, num_features_xd*2)
        self.drug2_fc_g2 = torch.nn.Linear(num_features_xd*2, output_dim)


        # DL cell featrues
        self.reduction = nn.Sequential(
            nn.Linear(num_features_xt, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
        self.fcllm = nn.Linear(3072, 768)

        # combined layers
        self.fc1 = nn.Linear(3*output_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, self.n_output)
        self.head = FusionHead()

    def forward(self, data1, data2):
        x1, edge_index1, batch1, cell = data1.x, data1.edge_index, data1.batch, data1.cell
        x2, edge_index2, batch2 = data2.x, data2.edge_index, data2.batch
         
        llmx1 = data1.llmx
        llmx2 = data2.llmx
        llmcell = data1.llmcell
        # print(llmx1.shape)

        # deal drug1
        x1 = self.drug1_conv1(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv2(x1, edge_index1)
        x1 = self.relu(x1)

        x1 = self.drug1_conv3(x1, edge_index1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)       # global max pooling

        # flatten
        x1 = self.relu(self.drug1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.drug1_fc_g2(x1)
        x1 = self.dropout(x1)                                                                                               
        # print('x1.shape', x1.shape)
        # print('x1', x1[0])


        # deal drug2
        x2 = self.drug1_conv1(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug1_conv2(x2, edge_index2)
        x2 = self.relu(x2)

        x2 = self.drug1_conv3(x2, edge_index2)
        x2 = self.relu(x2)
        x2 = gmp(x2, batch2)  # global max pooling

        # flatten
        x2 = self.relu(self.drug1_fc_g1(x2))
        x2 = self.dropout(x2)
        x2 = self.drug1_fc_g2(x2)
        x2 = self.dropout(x2)
        # print('x2.shape', x2.shape)
        # print('x', x2[0])

        # deal cell
        cell_vector = F.normalize(cell, 2, 1)
        cell_vector = self.reduction(cell_vector)
        
        #LLM
        llmx1 = self.fcllm(llmx1)
        llmx2 = self.fcllm(llmx2)
        llmcell = self.fcllm(llmcell)
        # print(x1.shape)
        # print(llmx1.shape)
        all_features_llm = torch.cat((llmx1,llmx2,llmcell),1)
        outllm = self.head(all_features_llm)
        
        
        contrastive_loss = contrastive_loss_v2(x1, llmx1) + contrastive_loss_v2(x2, llmx2) + contrastive_loss_v2(cell_vector, llmcell)
        
     
        
        # concat
        xc = torch.cat((x1, x2, cell_vector), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out,outllm,contrastive_loss
