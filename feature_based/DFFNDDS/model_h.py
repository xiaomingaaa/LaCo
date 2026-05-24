import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Sequential, Linear, ReLU
from contrastive import contrastive_loss_v2
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sklearn.metrics
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,roc_auc_score#改成分类
from head import FusionHead

CHAR_SMI_SET_LEN = 64


def random_feature_masking(x, mask_ratio=0.8):
    """
    x: [..., D] 只在最后一维做mask
    """
    D = x.size(-1)
    num_mask = int(D * mask_ratio)
    if num_mask <= 0:
        return x

    x_flat = x.reshape(-1, D)          # [N, D]
    idx = torch.randperm(D, device=x.device)[:num_mask]
    x_flat[:, idx] = 0                 # 直接置零（比 mask 乘法更快）
    return x_flat.reshape_as(x)
    


# def random_feature_masking(x, mask_ratio=0.6, per_sample=True):
#     """
#     x: [..., D] 只在最后一维做mask
#     per_sample=True: 每个样本/位置独立mask（更强）
#     """
#     if mask_ratio is None or mask_ratio <= 0:
#         return x

#     x = x.clone()  # 避免原地修改导致缓存/后续计算被污染

#     if per_sample:
#         # 每个元素独立随机mask（Bernoulli dropout式）
#         keep = (torch.rand_like(x) > mask_ratio).to(x.dtype)
#         return x * keep
#     else:
#         # 全batch共享同一组mask维度
#         D = x.size(-1)
#         num_mask = int(D * mask_ratio)
#         if num_mask <= 0:
#             return x
#         idx = torch.randperm(D, device=x.device)[:num_mask]
#         x[..., idx] = 0
#         return x


class ResDilaCNNBlock(nn.Module):
    def __init__(self, dilaSize, filterSize=64, dropout=0.15, name='ResDilaCNNBlock'):
        super(ResDilaCNNBlock, self).__init__()
        self.layers = nn.Sequential(
                        # LayerNormChannels(filterSize),
                        nn.ReLU(),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                        nn.ReLU(),
                        nn.Conv1d(filterSize,filterSize,kernel_size=3,padding=dilaSize,dilation=dilaSize),
                     )
        self.name = name
    def forward(self, x):
        # x: batchSize × filterSize × seqLen
        return x + self.layers(x)

class ResDilaCNNBlocks(nn.Module):
    def __init__(self, feaSize, filterSize, blockNum=5, dilaSizeList=[1,2,4,8,16], dropout=0.15, name='ResDilaCNNBlocks'):
        super(ResDilaCNNBlocks, self).__init__()
        self.blockLayers = nn.Sequential()
        self.linear = nn.Linear(feaSize,filterSize)
        for i in range(blockNum):
            self.blockLayers.add_module(f"ResDilaCNNBlock{i}", ResDilaCNNBlock(dilaSizeList[i%len(dilaSizeList)],filterSize,dropout=dropout))
        self.name = name

    def forward(self, x):
        # x: batchSize × seqLen × feaSize
        x = self.linear(x) # => batchSize × seqLen × filterSize
        x = self.blockLayers(x.transpose(1,2)) # => batchSize × seqLen × filterSize
        x = F.relu(x) # => batchSize × seqLen × filterSize
        x = x.transpose(1,2)
        x, _ = torch.max(x, 1)
        return x 


class MultiHeadAttentionInteract(nn.Module):
    """
        多头注意力的交互层
    """

    def __init__(self, embed_size, head_num, dropout, residual=True):
        """
        """
        super(MultiHeadAttentionInteract, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num
        self.dropout = dropout
        self.use_residual = residual
        self.attention_head_size = embed_size // head_num

        # 直接定义参数, 更加直观
        self.W_Q = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_K = nn.Parameter(torch.Tensor(embed_size, embed_size))
        self.W_V = nn.Parameter(torch.Tensor(embed_size, embed_size))

        if self.use_residual:
            self.W_R = nn.Parameter(torch.Tensor(embed_size, embed_size))

        # 初始化, 避免计算得到nan
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight)

    def forward(self, x):
        """
            x : (batch_size, feature_fields, embed_dim)
        """

        # 线性变换到注意力空间中
        Query = torch.tensordot(x, self.W_Q, dims=([-1], [0]))
        Key = torch.tensordot(x, self.W_K, dims=([-1], [0]))
        Value = torch.tensordot(x, self.W_V, dims=([-1], [0]))

        # Head (head_num, bs, fields, D / head_num)
        Query = torch.stack(torch.split(Query, self.attention_head_size, dim=2))
        Key = torch.stack(torch.split(Key, self.attention_head_size, dim=2))
        Value = torch.stack(torch.split(Value, self.attention_head_size, dim=2))

        # 计算内积
        inner = torch.matmul(Query, Key.transpose(-2, -1))
        inner = inner / self.attention_head_size ** 0.5

        # Softmax归一化权重
        attn_w = F.softmax(inner, dim=-1)
        #         attn_w = entmax15(inner, dim=-1)

        attn_w = F.dropout(attn_w, p=self.dropout)

        # 加权求和
        results = torch.matmul(attn_w, Value)

        # 拼接多头空间
        results = torch.cat(torch.split(results, 1, ), dim=-1)
        results = torch.squeeze(results, dim=0)  # (bs, fields, D)

        # 残差学习
        if self.use_residual:
            results = results + torch.tensordot(x, self.W_R, dims=([-1], [0]))

        results = F.relu(results)

        return results



class Highway(nn.Module):
    r"""Highway Layers
    Args:
        - num_highway_layers(int): number of highway layers.
        - input_size(int): size of highway input.
    """

    def __init__(self, num_highway_layers, input_size):
        super(Highway, self).__init__()
        self.num_highway_layers = num_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.linear = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.gate = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.num_highway_layers)])
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in range(self.num_highway_layers):
            gate = torch.sigmoid(self.gate[layer](x))
            # Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))
            # Compute non linear information
            linear = self.linear[layer](x)
            # Compute linear information
            x = gate * non_linear + (1 - gate) * linear
            # Combine non linear and linear information according to gate
            x = self.dropout(x)
        return x


class DualInteract(nn.Module):

    def __init__(self, field_dim, embed_size, head_num, dropout=0.5):
        super(DualInteract, self).__init__()

        self.bit_wise_net = Highway(input_size=field_dim * embed_size,
                                    num_highway_layers=2)

        hidden_dim = 1024

        self.vec_wise_net = MultiHeadAttentionInteract(embed_size=embed_size,#128
                                                       head_num=head_num,#8
                                                       dropout=dropout)

        self.trans_bit_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout)
        )
        self.trans_vec_nn = nn.Sequential(
            nn.LayerNorm(field_dim * embed_size),
            nn.Linear(field_dim * embed_size, hidden_dim),
            # nn.ReLU(),
            nn.GELU(),
            nn.Linear(hidden_dim, field_dim * embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
            x : batch, field_dim, embed_dim
        """

        b, f, e = x.shape
        bit_wise_x = self.bit_wise_net(x.reshape(b, f * e))
        vec_wise_x = self.vec_wise_net(x).reshape(b, f * e)

        m_bit = self.trans_bit_nn(bit_wise_x)
        m_vec = self.trans_vec_nn(vec_wise_x)
        m_x = m_vec + m_bit + x.reshape(b, f * e)
        # m_x = m_vec + x.reshape(b,f * e)
        return m_x


class MultiViewNet(nn.Module):

    def __init__(self, embed_dim=384,
                 mask_ratio_smi=0.8,
                 mask_ratio_fp=0.6,
                 mask_ratio_ctx=0.6,
                 per_sample_mask=True,
                 mask_llm=False,
                 mask_ratio_llm=0.6):
        super(MultiViewNet, self).__init__()

        hidden_dim = 1024
        proj_dim = 768
        dropout_rate = 0.5

        # ====== Mask超参 ======
        self.mask_ratio_smi = mask_ratio_smi
        self.mask_ratio_fp = mask_ratio_fp
        self.mask_ratio_ctx = mask_ratio_ctx
        self.per_sample_mask = per_sample_mask

        self.mask_llm = mask_llm
        self.mask_ratio_llm = mask_ratio_llm

        # ====== 原有网络 ======
        self.feature_interact = DualInteract(field_dim=5, embed_size=proj_dim, head_num=4)

        self.projection_smi_1 = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_smi_2 = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_context = nn.Sequential(
            nn.Linear(18046, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_fp1 = nn.Sequential(
            nn.Linear(1024, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )
        self.projection_fp2 = nn.Sequential(
            nn.Linear(1024, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout_rate),
        )

        self.transform = nn.Sequential(
            nn.LayerNorm(proj_dim * 5),
            nn.Linear(proj_dim * 5, 2),
        )

        self.align = nn.Sequential(
            nn.LayerNorm(proj_dim * 2),
            nn.Linear(proj_dim * 2, 1),
            torch.nn.Sigmoid()
        )

        self.fc = nn.Linear(3072, 768)
        self.head = FusionHead()

        self.norm = nn.LayerNorm(proj_dim * 5)

    def forward(self, smile_1_vectors, smile_2_vectors, context, fp1_vectors, fp2_vectors,
                LLMdruga, LLMdrugb, LLMcell):

        # ============================================================
        # 关键改动：在 projection 之前，对“原始输入”做 mask
        # 只在训练时启用（eval/test 不mask）
        # ============================================================
        if self.training:
            # SMILES embedding: [B, embed_dim]
            # smile_1_vectors = random_feature_masking(
            #     smile_1_vectors, self.mask_ratio_smi, per_sample=self.per_sample_mask
            # )
            # smile_2_vectors = random_feature_masking(
            #     smile_2_vectors, self.mask_ratio_smi, per_sample=self.per_sample_mask
            # )

            # FP: [B, 1024]
            # fp1_vectors = random_feature_masking(
            #     fp1_vectors, self.mask_ratio_fp, per_sample=self.per_sample_mask
            # )
            # fp2_vectors = random_feature_masking(
            #     fp2_vectors, self.mask_ratio_fp, per_sample=self.per_sample_mask
            # )

            # context: 你的dataset输出是 torch.FloatTensor([context_features])
            # DataLoader后通常是 [B, 1, 18046]，mask函数支持这种形状
            # context = random_feature_masking(
            #     context, self.mask_ratio_ctx, per_sample=self.per_sample_mask
            # )

            # （可选）LLM特征也mask：不建议默认开启，因为会影响对比学习对齐稳定性
            if self.mask_llm:
                LLMdruga = random_feature_masking(LLMdruga, self.mask_ratio_llm, per_sample=self.per_sample_mask)
                LLMdrugb = random_feature_masking(LLMdrugb, self.mask_ratio_llm, per_sample=self.per_sample_mask)
                LLMcell  = random_feature_masking(LLMcell,  self.mask_ratio_llm, per_sample=self.per_sample_mask)
                
            # if torch.rand(1).item() < 0.01:
            #     with torch.no_grad():
            #         smi1_zero = (smile_1_vectors == 0).float().mean().item()
            #         smi2_zero = (smile_2_vectors == 0).float().mean().item()
            #         fp1_zero = (fp1_vectors == 0).float().mean().item()
            #         fp2_zero = (fp2_vectors == 0).float().mean().item()
            #         ctx_zero = (context == 0).float().mean().item()
            #         print(
            #             f"[mask debug] "
            #             f"ratio_smi={self.mask_ratio_smi}, "
            #             f"ratio_fp={self.mask_ratio_fp}, "
            #             f"ratio_ctx={self.mask_ratio_ctx} | "
            #             f"smi1_zero={smi1_zero:.3f}, smi2_zero={smi2_zero:.3f}, "
            #             f"fp1_zero={fp1_zero:.3f}, fp2_zero={fp2_zero:.3f}, "
            #             f"ctx_zero={ctx_zero:.3f}"
            #         )

        # ====== 下面保持你原来的逻辑（projection之后再融合/分类） ======
        smile_1_vectors = self.projection_smi_1(smile_1_vectors)
        smile_2_vectors = self.projection_smi_2(smile_2_vectors)
        contextFeatures = self.projection_context(context)
        fp1_vectors = self.projection_fp1(fp1_vectors)
        fp2_vectors = self.projection_fp2(fp2_vectors)
        
        # smile_1_vectors = random_feature_masking(
        #         smile_1_vectors, self.mask_ratio_smi)
        # smile_2_vectors = random_feature_masking(
        #         smile_2_vectors, self.mask_ratio_smi)
        # contextFeatures = random_feature_masking(
        #         contextFeatures, self.mask_ratio_smi)
        # fp1_vectors = random_feature_masking(
        #         fp1_vectors, self.mask_ratio_smi)
        # fp2_vectors = random_feature_masking(
        #         fp2_vectors, self.mask_ratio_smi)

        # LLM降维
        LLMdruga = F.relu(self.fc(LLMdruga))
        LLMdrugb = F.relu(self.fc(LLMdrugb))
        LLMcell  = F.relu(self.fc(LLMcell))
        
#         print("drugA_embed:", smile_1_vectors.shape)
#         print("drugB_embed:", smile_2_vectors.shape)
#         print("cell_embed:", contextFeatures.shape)

#         print("LLM_drugA:", LLMdruga.shape)
#         print("LLM_drugB:", LLMdrugb.shape)
#         print("LLM_cell:", LLMcell.shape)


        # 对比损失
        contrastive_loss = (
            contrastive_loss_v2(smile_1_vectors, LLMdruga)
            + contrastive_loss_v2(smile_2_vectors, LLMdrugb)
            +contrastive_loss_v2(fp1_vectors, LLMdruga)
            + contrastive_loss_v2(fp2_vectors, LLMdrugb)
            + contrastive_loss_v2(contextFeatures.squeeze(1), LLMcell)
        )

        # 多模态堆叠 -> 交互 -> 分类
        all_features = torch.stack(
            [smile_1_vectors, smile_2_vectors, contextFeatures.squeeze(1), fp1_vectors, fp2_vectors],
            dim=1
        )
        all_features2 = torch.stack(
            [LLMdruga, LLMdrugb, LLMcell, LLMdruga, LLMdruga],
            dim=1
        )
        all_features = self.feature_interact(all_features)
        all_features2 = self.feature_interact(all_features2)
        out1 = self.transform(all_features)
        out2 = self.transform(all_features2)

        return out1, out2,contrastive_loss


class DecoderNet(nn.Module):

    def __init__(self, embed_dim=768):
        super(DecoderNet, self).__init__()
        hidden_dim = 1024
        proj_dim = 128  # 128

        self.feature_interact = DualInteract(field_dim=3, embed_size=proj_dim, head_num=8)  # 修改处 dim=3改成5

        self.projection_smi_1 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim)
        )
        self.projection_smi_2 = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, proj_dim)
        )
        self.projection_context = nn.Sequential(
            nn.LayerNorm(112),  # 改变norm的维度
            nn.Linear(112, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )
        # self.projection_fp1 = nn.Sequential(
        #     nn.LayerNorm(1024),
        #     nn.Linear(1024, proj_dim)
        # )
        # self.projection_fp2 = nn.Sequential(
        #     nn.LayerNorm(1024),
        #     nn.Linear(1024, proj_dim)
        # )
        self.transform = nn.Sequential(
            nn.LayerNorm(proj_dim * 3),
            nn.Linear(proj_dim * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 384),
            nn.ReLU(),
            nn.Linear(384, 1),
            torch.nn.Sigmoid(),  # 512全改为384 需要输出为一维
        )
        self.decoder_smi_1 = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 768)
        )
        self.decoder_smi_2 = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Linear(proj_dim, 768)
        )
        self.decoder_context = nn.Sequential(
            nn.LayerNorm(proj_dim),  # 改变norm的维度
            nn.Linear(proj_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 112),
        )

        self.norm = nn.LayerNorm(proj_dim * 3)  # 为什么是四层 proj_dim 原本为3

    def forward(self, smile_1_vectors, smile_2_vectors, context):
        smile_1_vectors = self.projection_smi_1(smile_1_vectors)
        smile_2_vectors = self.projection_smi_2(smile_2_vectors)
        contextFeatures = self.projection_context(context)  # contextfeatures能不能reshape#尝试reshape
        # fp1_vectors = self.projection_fp1(fp1_vectors)
        # fp2_vectors = self.projection_fp2(fp2_vectors)
        smile_1_decode   = self.decoder_smi_1(smile_1_vectors)
        smile_2_decode   = self.decoder_smi_2(smile_2_vectors)
        context_decode   = self.decoder_context(contextFeatures)
         # 改变维度  试一下
        all_features = torch.stack([smile_1_vectors, smile_2_vectors, contextFeatures.squeeze(1)],dim=1)  # 改变维度  试一下
        all_features = self.feature_interact(all_features)
        print(all_features.shape)
        out = self.transform(all_features)

        return out,smile_1_decode,smile_2_decode,context_decode


def test(model: nn.Module, test_loader, loss_function, device, show):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)

            y_hat = model(*x)

            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)

    test_loss /= len(test_loader.dataset)

    evaluation = {
        'loss': test_loss,
        'accuracy_score': accuracy_score.score(targets, outputs),
        'confusion': confusion_matrix(targets, outputs),
        'report': classification_report(targets, outputs),
    }

    return evaluation
