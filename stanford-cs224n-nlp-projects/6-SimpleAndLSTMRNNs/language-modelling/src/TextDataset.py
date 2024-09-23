from torch.utils.data import Dataset
import spacy
import torch
import re

nlp = spacy.load("en_core_web_sm")

class TextDataset(Dataset):
    def __init__(self, 
                 file_name,
                 seq_len
                 ):
        with open(file_name, 'r', encoding='utf-8') as f:
            self.text = f.read().lower()

        self.seq_len = seq_len
        self.tokens = self.tokenize(self.text)
        self.token_2_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_2_idx.items()}
        self.vocab_size = len(self.token_2_idx)

        # Convert the tokenized words into corresponding indices
        self.data = [self.token_2_idx[token] for token in self.tokens]
    
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


# with open('data.txt', 'r', encoding='utf-8') as f:
#     text = f.read().lower()

# doc = nlp(text)
# tokens = [token.text for token in doc if not token.is_space]
# print(len(tokens))
# print(len(set(tokens)))
# print(len(sorted(set(tokens))))