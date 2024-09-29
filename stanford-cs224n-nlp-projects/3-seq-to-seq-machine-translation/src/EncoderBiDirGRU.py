import torch.nn as nn
import torch

class EncoderBiDirGRU(nn.Module):
    def __init__(self,
                 src_vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float):
        super(EncoderBiDirGRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(src_vocab_size, embed_dim)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Linear layer to reduce the concatenated hidden state size from bidirectional
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch_size, src_len]

        # Embedding the input sequence
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, embed_dim]

        # Pass through the bidirectional GRU
        outputs, hidden = self.gru(embedded)  # outputs: [batch_size, src_len, hidden_dim * 2], hidden: [num_layers * 2, batch_size, hidden_dim]

        # Separate forward and backward hidden states
        forward_hidden = hidden[0:hidden.size(0):2]  # Take even-indexed hidden states (forward)
        backward_hidden = hidden[1:hidden.size(0):2]  # Take odd-indexed hidden states (backward)

        # Merge the forward and backward hidden states by concatenating along the hidden_dim
        merged_hidden = torch.cat((forward_hidden, backward_hidden), dim=2)  # [num_layers, batch_size, hidden_dim * 2]
        merged_hidden = self.fc(merged_hidden)  # [num_layers, batch_size, hidden_dim]

        return outputs, merged_hidden
