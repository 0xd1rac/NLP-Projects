import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentLSTM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 is_bidirectional: bool,
                 output_size: int = 1,
                 dropout: float = 0.5):
        """
        Sentiment LSTM model.

        Args:
            vocab_size (int): Number of words in the vocabulary.
            embedding_dim (int): Dimension of the embedding layer.
            hidden_dim (int): Number of hidden units in the LSTM.
            num_layers (int): Number of LSTM layers.
            is_bidirectional (bool): Whether to use a bidirectional LSTM.
            output_size (int): The number of output classes (for binary sentiment classification, this is 1).
            dropout (float): Dropout probability.
        """
        super(SentimentLSTM, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            bidirectional=is_bidirectional, 
                            dropout=dropout if num_layers > 1 else 0, 
                            batch_first=True)

        # Dimensions of the fully connected layer depend on whether LSTM is bidirectional
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

        # LSTM output
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # If using a bidirectional LSTM, we concatenate the hidden states from both directions
        if self.is_bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]  # Use the last hidden state from the LSTM

        # Apply dropout to the hidden layer
        hidden = self.dropout(hidden)

        # Pass the last hidden state through the fully connected layer
        output = self.fc(hidden)

        return output


