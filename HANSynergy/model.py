import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv


def contrastive_loss_v2(feat, llm_feat, temperature=0.07, hard_negative=False):
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


class MultiHeadAttentionInteract(nn.Module):

    def __init__(self, embed_size, head_num, dropout, residual=True):
        super(MultiHeadAttentionInteract, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num
        self.W_Q = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_K = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_V = nn.Parameter(torch.Tensor(embed_size, embed_size))
        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(embed_size, embed_size))
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))
        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))
        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5
        attn_w = F.softmax(inner, dim=-1)
        attn_w = F.dropout(attn_w, p=self.dropout)
        results = torch.matmul(attn_w, Value)
        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)
        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))
        results = F.relu(results)
        return results


class FeatureFusion(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout=0.5):
        super(FeatureFusion, self).__init__()
        hidden_dim = 1024
        self.vec_wise_net = MultiHeadAttentionInteract(embed_size=embed_size,  # 128
                                                       head_num=head_num,  # 8
                                                       dropout=dropout)
        self.trans_vec_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """
        b, f, e = x.shape
        vec_wise_x = self.vec_wise_net(x).reshape(b, f * e)
        m_vec = self.trans_vec_nn(vec_wise_x)
        m_x = F.relu(m_vec) + x.reshape(b, f * e)
        return m_x


class ResBlock(nn.Module):

    """
    Initialize a residual block with two convolutions followed by batchnorm layers
    """

    def __init__(self, in_size: int, hidden_size: int, out_size: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, out_size, 3, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = nn.BatchNorm1d(out_size)

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x

    """
    Combine output with the original input
    """

    def forward(self, x):
        return x + self.convblock(x)  # skip connection


class FeatureFusionAndPredictModel(nn.Module):

    def __init__(self, embed_dim=384, field_dim=5, dropout_rate=0.5, batch_size=256):
        super(FeatureFusionAndPredictModel, self).__init__()
        proj_dim = 256
        self.feature_interact = FeatureFusion(field_dim=field_dim, embed_size=proj_dim, head_num=2)
        self.proj_dim = proj_dim
        self.field_dim = field_dim
        self.batch_size = batch_size
        self.smile1_map = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.smile2_map = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.cell_line_map = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.fp1_map = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.fp2_map = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.transform = nn.Sequential(
            nn.LayerNorm(proj_dim*field_dim),
            nn.Linear(proj_dim*field_dim, 2),
        )
        self.resblock1 = ResBlock(
            in_size=field_dim,
            hidden_size=field_dim,
            out_size=field_dim
        )
        self.resblock2 = ResBlock(
            in_size=field_dim,
            hidden_size=field_dim,
            out_size=field_dim
        )
        self.resblock3 = ResBlock(
            in_size=field_dim,
            hidden_size=field_dim,
            out_size=field_dim
        )
        self.drug1_pre_map = nn.Identity()
        self.drug2_pre_map = nn.Identity()
        self.cell_pre_map  = nn.Identity()


        
        

    def forward(self, smile1_feature, smile2_feature, cell_line_feature, smile1_fp_feature, smile2_fp_feature):
        smile1_feature = self.smile1_map(smile1_feature)
        smile2_feature = self.smile2_map(smile2_feature)
        cell_line_feature = self.cell_line_map(cell_line_feature)
        smile1_fp_feature = self.fp1_map(smile1_fp_feature)
        smile2_fp_feature = self.fp2_map(smile2_fp_feature)
        features = torch.stack([
        smile1_fp_feature, smile1_feature,
        smile2_fp_feature, smile2_feature,
        cell_line_feature
        ], dim=1)
        features = self.feature_interact(features)
        features = features.reshape([self.batch_size, self.field_dim, -1])
        features = self.resblock1(features)
        features = self.resblock2(features)
        features = self.resblock3(features)
        features = features.reshape([-1, self.field_dim*self.proj_dim])
        out = self.transform(features)
        return out


class HGANDDS(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        hidden_channels: int,
        drug_feature_length: int = 384,
        protein_feature_length: int = 384,
        cell_feature_length: int = 18046,
        dropout_rate: float = 0.5,
        is_gnn: bool = True,
        data_type: str = 'drugcombdb'
    ) -> None:
        super().__init__()
        self.data_type = data_type
        self.drug_map = nn.Sequential(
            nn.Linear(drug_feature_length, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout_rate),
        )
        self.drug_fp_map = nn.Sequential(
            nn.Linear(1024, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout_rate),
        )
        self.protein_map = nn.Sequential(
            nn.Linear(protein_feature_length, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout_rate),
        )
        self.cell_map = nn.Sequential(
            nn.Linear(cell_feature_length, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout_rate),
        )
        
        self.drug_pretrained_map = nn.Sequential(
            nn.Linear(3072, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout_rate),
        ) 

        self.cell_pretrained_map = nn.Sequential(
            nn.Linear(3072, hidden_channels),
            nn.ReLU(),
            nn.LayerNorm(hidden_channels),
            nn.Dropout(dropout_rate),
        )

        self.tissue_emb = nn.Embedding(torch.max(data['tissue'].node_id)+1, hidden_channels)
        self.is_gnn = is_gnn
        if self.is_gnn:
            self.gnn = GNN(
                hidden_channels=hidden_channels
            )
            self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        else:
            self.han = HAN(
                hidden_channels=hidden_channels,
                data=data,
            )
        self.feature_model = FeatureFusionAndPredictModel(
            embed_dim=hidden_channels,
            field_dim=5
        )
        
        self.llm_proj_drug = nn.Linear(3072, hidden_channels)
        self.llm_proj_cell = nn.Linear(3072, hidden_channels)


    def forward(
        self,
        data: HeteroData,
        cid_nodeid_list1, cid_nodeid_list2, cell_node_list,
            drug1_embed, drug2_embed, cell_embed
    ):
        x_dict = {
            'drug': self.drug_map(data['drug'].feature),
            'drug_fp': self.drug_fp_map(data['drug_fp'].feature),
            'cell': self.cell_map(data['cell'].feature),
            'protein': self.protein_map(data['protein'].feature),
            'tissue': self.tissue_emb(data['tissue'].node_id)
        }
        if self.is_gnn:
            x_dict = self.gnn(x_dict, data.edge_index_dict)
        else:
            x_dict = self.han(x_dict, data.edge_index_dict)
            
        drug1_pre = self.drug_pretrained_map(drug1_embed)
        drug2_pre = self.drug_pretrained_map(drug2_embed)
        cell_pre  = self.cell_pretrained_map(cell_embed)
        
        drug1_feature = x_dict['drug'][cid_nodeid_list1]
        drug2_feature = x_dict['drug'][cid_nodeid_list2]
        fp1_feature = x_dict['drug_fp'][cid_nodeid_list1]
        fp2_feature = x_dict['drug_fp'][cid_nodeid_list2]
        cell_feature = x_dict['cell'][cell_node_list]
        
        drug1_embed_proj = self.llm_proj_drug(drug1_embed)
        drug2_embed_proj = self.llm_proj_drug(drug2_embed)
        cell_embed_proj  = self.llm_proj_cell(cell_embed)

        contrastive_loss = (
            contrastive_loss_v2(drug1_feature, drug1_embed_proj) +
            contrastive_loss_v2(drug2_feature, drug2_embed_proj) +
            contrastive_loss_v2(cell_feature, cell_embed_proj)
           )

        out = self.feature_model(drug1_feature, drug2_feature, cell_feature, fp1_feature, fp2_feature)
        return out,contrastive_loss


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        # self.conv3 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x, edge_index))
        # x = F.relu(self.conv2(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class HAN(torch.nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        data: HeteroData,
    ) -> None:
        super().__init__()
        self.conv1 = HANConv(hidden_channels, hidden_channels, data.metadata(), heads=4)
        self.conv2 = HANConv(hidden_channels, hidden_channels, data.metadata(), heads=4)
        # self.conv3 = HANConv(hidden_channels, hidden_channels, data.metadata(), heads=4)

    def forward(self, x: torch.Tensor, edge_index_dict: torch.Tensor, return_semantic_attention_weights=False):
        x = self.conv1(x, edge_index_dict, return_semantic_attention_weights)
        x = self.conv2(x, edge_index_dict, return_semantic_attention_weights)
        # x = self.conv3(x, edge_index_dict, return_semantic_attention_weights)
        return x
