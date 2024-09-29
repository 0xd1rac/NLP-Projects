import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.decoder_projection = nn.Linear(hidden_dim, hidden_dim * 2)  # Project decoder hidden to match encoder hidden

    def forward(self, encoder_outputs, decoder_hidden_state):
        """
        encoder_outputs: (batch_size, seq_len, hidden_dim * 2) -> bidirectional GRU outputs for all timesteps
        decoder_hidden_state: (batch_size, hidden_dim) -> last hidden state from the decoder

        Returns:
        context_vector: (batch_size, hidden_dim * 2) -> context vector for the current decoder step
        attention_weights: (batch_size, seq_len) -> attention weights for each timestep in the encoder outputs
        """
        # Project decoder hidden state to match encoder hidden state dimensionality
        decoder_hidden_state = self.decoder_projection(decoder_hidden_state)  # (batch_size, hidden_dim * 2)

        # Add a time step dimension to decoder hidden state
        decoder_hidden_state = decoder_hidden_state.unsqueeze(1)  # (batch_size, 1, hidden_dim * 2)

        # Compute attention scores using dot product
        attention_scores = torch.bmm(encoder_outputs, decoder_hidden_state.transpose(1, 2)).squeeze(2)
        # attention_scores: (batch_size, seq_len)

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)

        # Compute the context vector as the weighted sum of the encoder outputs
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_dim * 2)

        return context_vector, attention_weights
