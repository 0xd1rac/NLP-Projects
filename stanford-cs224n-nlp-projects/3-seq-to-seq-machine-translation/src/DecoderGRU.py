import torch.nn as nn

class DecoderGRU(nn.Module):
    def __init__(self,
                 target_vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float):
        super(DecoderGRU, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(target_vocab_size, embed_dim)

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.target_vocab_size = target_vocab_size

    def forward(self, input, hidden):
        """
        input: [batch_size] -> current input token ids
        hidden: [num_layers, batch_size, hidden_dim] -> previous hidden state
        """

        # Embedding the input token
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, embed_dim]

        # Pass through the GRU
        output, hidden = self.gru(embedded, hidden)  # output: [batch_size, 1, hidden_dim], hidden: [num_layers, batch_size, hidden_dim]

        # Generate output token logits
        output = self.fc_out(output.squeeze(1))  # [batch_size, target_vocab_size]

        return output, hidden
