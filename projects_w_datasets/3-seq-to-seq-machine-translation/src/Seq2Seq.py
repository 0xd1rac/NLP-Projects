import torch.nn as nn
import torch
from .DecoderGRU import DecoderGRU
from .EncoderBiDirGRU import EncoderBiDirGRU
from .Attention import Attention

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder: EncoderBiDirGRU, 
                 decoder: DecoderGRU, 
                 start_token_idx: int,
                 eos_token_idx: int,
                 device: torch.device,
                 use_attention: bool):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device 
        self.start_token_idx = start_token_idx
        self.eos_token_idx = eos_token_idx
        self.use_attention = use_attention

        if self.use_attention:
            self.attention = Attention(hidden_dim=decoder.hidden_dim)
            # Update this to match hidden_dim * 3 because of concatenation
            self.attention_fcn = nn.Linear(decoder.hidden_dim * 3, decoder.target_vocab_size)

    def forward(self, 
                src_seq,  # Source language sequence
                tgt_seq=None,  # Target language sequence (optional, used during training)
                teacher_forcing_ratio=0.5,  # Teacher forcing ratio for training
                max_len=20):
        # src_seq: [batch_size, src_len]
        # tgt_seq: [batch_size, tgt_len]  # optional
        batch_size = src_seq.size(0)
        tgt_vocab_size = self.decoder.target_vocab_size

        # Pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(src_seq)
        # decoder_hidden: Initialize decoder hidden state with encoder hidden state
        decoder_hidden = encoder_hidden

        # Check if we are in training mode or inference mode
        if tgt_seq is not None:
            # Training mode (with teacher forcing)
            tgt_len = tgt_seq.size(1)
            outputs = torch.zeros(tgt_len, batch_size, tgt_vocab_size).to(self.device)
            input = tgt_seq[:, 0]  # Start with the <start> token

            # Decode step by step
            for t in range(1, tgt_len):
                # Decode without attention for now
                output, decoder_hidden = self.decoder(input, decoder_hidden)

                if self.use_attention:
                    # Compute attention over all encoder outputs
                    context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden[-1])
                    
                    # Concatenate context vector with decoder hidden state
                    concat_output = torch.cat((context_vector, decoder_hidden[-1]), dim=1)  # (batch_size, hidden_dim * 3)
                    
                    # Pass through a fully connected layer
                    output = self.attention_fcn(concat_output)

                outputs[t] = output.squeeze(1)

                # Decide whether to use teacher forcing or not
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                top1 = output.argmax(-1)  # Get the predicted token

                # Use the ground truth token or the model's predicted token as the next input
                input = tgt_seq[:, t] if teacher_force else top1

        else:
            # Inference mode (no teacher forcing)
            outputs = torch.zeros(max_len, batch_size, tgt_vocab_size).to(self.device)
            input = torch.tensor([self.start_token_idx] * batch_size).to(self.device)

            for t in range(max_len):
                # Decode without attention for now
                output, decoder_hidden = self.decoder(input, decoder_hidden)

                if self.use_attention:
                    # Compute attention over all encoder outputs
                    context_vector, attention_weights = self.attention(encoder_outputs, decoder_hidden[-1])
                    
                    # Concatenate context vector with decoder hidden state
                    concat_output = torch.cat((context_vector, decoder_hidden[-1]), dim=1)  # (batch_size, hidden_dim * 3)
                    
                    # Pass through a fully connected layer
                    output = self.attention_fcn(concat_output)

                outputs[t] = output.squeeze(1)

                # Get the predicted token with the highest probability
                top1 = output.argmax(-1)

                # Stop decoding if the <EOS> token generated
                if (top1 == self.eos_token_idx).all():
                    break

                input = top1
        
        return outputs.permute(1,0,2)
