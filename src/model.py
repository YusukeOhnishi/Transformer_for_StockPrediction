import numpy as np
import math
import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super().__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.pos_encoder = PositionalEncoding(d_model=feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)

    def init_weights(self):
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform(-0.1, 0.1)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = self.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x+self.pe[:np.shape(x)[0], :])


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = (-val_loss)
        if self.best_score is None:
            self.best_score = score
        elif score == self.patience:
            self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
