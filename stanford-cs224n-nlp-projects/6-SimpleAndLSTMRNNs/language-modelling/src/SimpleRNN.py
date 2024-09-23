import torch 
import torch.nn as nn

"""
RNN Work Flow:
    h_t = f(W_e * e_t + W_h * h_t-1 + b_h)

    e_t: input embedding vector at time, t
    h_t-1: input hidden unit vector from the previous time step, t-1
    b_h: bias term 
    W_e: weight matrix for embedding vector 
    W_h: weight matrix for hidden unit vector from previous time step

    
dim(e_t) is specified using embedding_dim
dim(h_t) is specified using hidden_dim
"""

class SimpleRNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int
                 ):
        super(SimpleRNN, self).__init__()

        # Embedding layer - convert token indices into dense vectors of size, embedding_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=1,
                          batch_first=True
                          )
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, 
                x: torch.Tensor
                ):
        # (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)

        # Initialize hidden state (1 layer, batch_size, hidden_dim)
        batch_size = x.size(0)
        init_hidden = torch.zeros(1, batch_size, self.hidden_dim).to(x.device)

        # RNN output
        # For SINGLE LAYER RNNS: 
            # all_hidden: hidden states for every timestep in the sequence
            # last_hidden: hidden state from only the last timestep of the sequence 
        all_hidden, last_hidden = self.rnn(embedded, init_hidden)
        # Use the last hidden state for classification (from the last timestep)
        output = self.fc(all_hidden[:,-1]) # (batch_size, output_dim)
        return output
