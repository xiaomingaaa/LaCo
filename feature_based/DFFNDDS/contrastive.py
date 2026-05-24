import torch
import torch.nn.functional as F



"""
def contrastive_loss_v2(feat, llm_feat, temperature=0.07):
    """
"""
    基于余弦相似度的对比损失函数。
    
    feat: 从模型输出的真实嵌入（如 drug_embed 或 cell_embed）
    llm_feat: 从大模型输出的嵌入（如 LLM_drugA 或 LLM_cell）
    temperature: 温度系数，用于调整嵌入相似度的尺度
    """
"""
    feat = F.normalize(feat, p=2, dim=1)  # 对特征进行L2归一化
    llm_feat = F.normalize(llm_feat, p=2, dim=1)  # 对LLM特征进行L2归一化
    
    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(feat, llm_feat.T)  # [batch_size, batch_size]
    
    # 构建正样本和负样本对
    mask = torch.eye(feat.size(0), device=feat.device).bool()  # 选择正样本的对角线
    positive_similarity = similarity_matrix[mask].view(feat.size(0), -1)
    negative_similarity = similarity_matrix[~mask].view(feat.size(0), -1)
    
    # 合并正负样本对的相似度
    logits = torch.cat([positive_similarity, negative_similarity], dim=1)
    logits /= temperature  # 温度缩放

    # 构建标签，正样本标签为0（因为我们希望最大化正样本的概率）
    labels = torch.zeros(feat.size(0), dtype=torch.long).to(feat.device)
    
    # 使用交叉熵计算损失
    loss = F.cross_entropy(logits, labels)
    
    return loss
  """  
    
import torch
import torch.nn.functional as F

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


def kl_loss(q, p):
        kl_pq = F.kl_div(torch.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        kl_qp = F.kl_div(torch.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        sym_kl_div = kl_pq + kl_qp
        return sym_kl_div