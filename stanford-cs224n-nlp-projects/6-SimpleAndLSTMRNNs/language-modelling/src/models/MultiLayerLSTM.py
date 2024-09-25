import torch.nn as nn
import torch

class MultiLayerLSTM(nn.Module):
    def __init__(self, 
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int
                 ):
        super(MultiLayerLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

         # LSTM layer - multiple layers can be specified with num_layers
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True)
        
        # Fully connected layer to map hidden state to output
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, 
                x: torch.Tensor,
                init_hidden: torch.Tensor = None
                )->torch.Tensor:
        
        embedded = self.embedding(x)
        batch_size = x.size(0)
        if init_hidden == None:
            init_hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device),  # hidden state
                           torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)   # cell state
                           ) 
        # LSTM output
        # all_hidden: hidden states for every timestep in the sequence
        # (h_n, c_n): hidden state and cell state from the last timestep of the sequence
        all_hidden, (h_n, c_n) = self.lstm(embedded, init_hidden)

        # Use the last hidden state (h_n) for classification (from the last timestep)
        output = self.fc(all_hidden[:, -1]) # (batch_size, vocab_size)
        return output
