import torch
import torch.nn.functional as F

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    loss = loss.mean()
    return loss

def alignment(x, y, alpha=2):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(alpha).mean() 

def contrastive_loss_v2(feat, llm_feat, temperature=0.07):
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