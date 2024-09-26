import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentRNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 is_bidirectional: bool,
                 output_size: int = 1,
                 dropout: float = 0.5):
        """
        Sentiment RNN model using vanilla RNN.

        Args:
            vocab_size (int): Number of words in the vocabulary.
            embedding_dim (int): Dimension of the embedding layer.
            hidden_dim (int): Number of hidden units in the RNN.
            num_layers (int): Number of RNN layers.
            is_bidirectional (bool): Whether to use a bidirectional RNN.
            output_size (int): The number of output classes (for binary sentiment classification, this is 1).
            dropout (float): Dropout probability.
        """
        super(SentimentRNN, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # RNN layer
        self.rnn = nn.RNN(input_size=embedding_dim, 
                          hidden_size=hidden_dim, 
                          num_layers=num_layers, 
                          bidirectional=is_bidirectional, 
                          dropout=dropout if num_layers > 1 else 0, 
                          batch_first=True)

        # Dimensions of the fully connected layer depend on whether RNN is bidirectional
        self.is_bidirectional = is_bidirectional
        self.fc = nn.Linear(hidden_dim * 2 if is_bidirectional else hidden_dim, output_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input sequence of token indices (batch_size, sequence_length)

        Returns:
            torch.Tensor: The output predictions (batch_size, output_size)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)

        # RNN output
        rnn_out, hidden = self.rnn(embedded)

        # If using a bidirectional RNN, we concatenate the hidden states from both directions
        if self.is_bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]  # Use the last hidden state from the RNN

        # Apply dropout to the hidden layer
        hidden = self.dropout(hidden)

        # Pass the last hidden state through the fully connected layer
        output = self.fc(hidden)

        return output
