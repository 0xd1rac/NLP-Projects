from torch.utils.data import Dataset
import spacy
import torch
import re
import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
nlp = spacy.load("en_core_web_sm")

class TextDataset(Dataset):
    def __init__(self, 
                 file_path,
                 seq_len
                 ):
        self.text = self.read_file(file_path)
        self.seq_len = seq_len
        self.tokens = self.tokenize(self.text)
        self.token_2_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_2_idx.items()}
        self.vocab_size = len(self.token_2_idx)

        # Convert the tokenized words into corresponding indices
        self.data = [self.token_2_idx[token] for token in self.tokens]
    
    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
        return text

    def tokenize(self, text):
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_space]
        tokens = sorted(set(tokens))
        return tokens
    
    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        input_seq = self.data[idx: idx + self.seq_len]
        target_seq = self.data[idx+1: idx + self.seq_len + 1]
        return torch.tensor(input_seq), torch.tensor(target_seq)
