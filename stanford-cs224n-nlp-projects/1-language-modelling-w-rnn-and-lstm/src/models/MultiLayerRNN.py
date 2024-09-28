import torch.nn as nn
import torch
class MultiLayerRNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int
                 ):
        
        super(MultiLayerRNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size=embedding_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_layers,
                          batch_first=True
                          )
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, 
                x: torch.Tensor,
                init_hidden: torch.Tensor=None
                ) -> torch.Tensor:
        embedded = self.embedding(x)
        batch_size = x.size(0)
        
        if init_hidden == None:
            init_hidden = torch.zeros(self.rnn.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        rnn_out, hidden = self.rnn(embedded, init_hidden)
        last_hidden = rnn_out[:, -1, :] 
        output = self.fc(last_hidden)
        return output

