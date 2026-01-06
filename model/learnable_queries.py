import torch
import torch.nn as nn

class LearnableQueries(nn.Module):
    def __init__(self, num_classes: int, d_text: int, freeze: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.d_text = d_text
        self.emb = nn.Embedding(num_classes, d_text) #[N,C]
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)  
        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, batch_size: int) -> torch.Tensor:
        text = self.emb.weight
        return text.unsqueeze(0).expand(batch_size, -1, -1).contiguous()  #[B,N,C]