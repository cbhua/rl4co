from typing import Tuple, Union

import torch
import torch.nn as nn


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class AdaptiveSequential(nn.Sequential):
    def forward(
        self, *inputs: Union[Tuple[torch.Tensor], torch.Tensor]
    ) -> Union[Tuple[torch.Tensor], torch.Tensor]:
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Normalization(nn.Module):
    def __init__(self, embed_dim, normalization="batch"):
        super(Normalization, self).__init__()
        if normalization != "layer":
            normalizer_class = {
                "batch": nn.BatchNorm1d,
                "instance": nn.InstanceNorm1d,
            }.get(normalization, None)

            self.normalizer = normalizer_class(embed_dim, affine=True)
        else:
            self.normalizer = "layer"

    def forward(self, x):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(x.view(-1, x.size(-1))).view(*x.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(x.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.normalizer == "layer":
            return (x - x.mean((1, 2)).view(-1, 1, 1)) / torch.sqrt(
                x.var((1, 2)).view(-1, 1, 1) + 1e-05
            )
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = embed_dim
        max_len = max_len
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(max_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, hidden: torch.Tensor, seq_pos) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
            seq_pos: Tensor, shape ``[batch_size, seq_len]``
        """
        pes = self.pe.expand(hidden.size(0), -1, -1).gather(
            1, seq_pos.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        hidden = hidden + pes
        return self.dropout(hidden)
