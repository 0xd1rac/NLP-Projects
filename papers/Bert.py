import torch.nn as nn
import torch 
import math
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEmbedding, self).__init__()

        pos_embed = torch.zeros(max_len, d_model).float()
        pos_embed.require_grad = False

        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pos_embed[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pos_embed[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        self.pos_embed = pos_embed.unsqueeze(0)

    def forward(self, x):
        return self.pos_embed

class BertEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_size,
                 seq_len,
                 dropout
                 ):
        super(BertEmbedding, self).__init__()
        self.embed_size = embed_size

        self.token_embed = torch.nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # segment 0: first sentence in the input pair
        # segment 1: second sentence in the input pair
        self.segment_embed = torch.nn.Embedding(2, embed_size, padding_idx=0)  
        self.pos_embed = PositionalEmbedding(d_model=embed_size, max_len=seq_len)

        self.dropout = nn.Dropout(dropout)

    
    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 num_heads,
                 d_model, 
                 dropout=0.1
                 ):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model needs to be divisible by num_heads"

        self.d_k = d_model // num_heads # dim per head
        self.num_heads = num_heads
        self.d_model = d_model

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_Q(query)
        K = self.W_K(key)
        V = self.W_V(value)

        # Split the heads: [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2,-1)) / self.scale # [batch_size, num_heads, seq_len, seq_len]

        # apply mask (if any)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = torch.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, V)  # [batch_size, num_heads, seq_len, d_k]
        # Concatenate heads: [batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Final linear projection to get the output: [batch_size, seq_len, d_model]
        output = self.out_proj(attention_output)

        return output, attention_weights


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.gelu(self.fc1(x))
        out = self.fc2(self.dropout(out))
        return out


class EncoderBlock(nn.Module):
    