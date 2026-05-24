import torch
import torch.nn as nn


class FusionHead(torch.nn.Module):
    def __init__(self, out_channels=256):
        super(FusionHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(768*3 , int(768)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(768), 384),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(384, 1),

            )
        
    def forward(self, out):
        # out = torch.cat((x_cell_embed, drug_embed), dim=1)
        out = self.fc(out)
        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

