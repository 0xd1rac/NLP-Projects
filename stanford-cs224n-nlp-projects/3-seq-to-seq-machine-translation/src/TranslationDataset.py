import csv
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import spacy
import json

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def read_csv_data(file_path):
    src_sentences = []
    tgt_sentences = []

    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) # skip the header row 
        for row in reader:
            # there are some rows where English entry wasn't filled
            if len(row) == 2:
                src_sentences.append(row[0]) # German source 
                tgt_sentences.append(row[1]) # English target 
    return src_sentences, tgt_sentences

# Vocab building function
def build_vocab(tokenized_sentences, min_freq=2):
    counter = Counter([token for sentence in tokenized_sentences for token in sentence])
    token_2_idx = {token: idx+2 for idx, (token, freq) in enumerate(counter.items()) if freq >= min_freq}
    token_2_idx['<pad>'] = 0 
    token_2_idx['<unk>'] = 1
    token_2_idx['<sos>'] = len(token_2_idx)
    token_2_idx['<eos>'] = len(token_2_idx)

    idx_2_token = {idx: token for token, idx in token_2_idx.items()}
    return token_2_idx, idx_2_token

# Convert tokenized sentence to tensor of word indices
def sentence_to_tensor(sentence, token_2_idx):
    tensor = []
    for token in sentence:
        idx = token_2_idx.get(token, token_2_idx['<unk>'])
        tensor.append(idx)
        # if idx == token_2_idx['<unk>']:
        #     print(f"Out-of-vocabulary token: {token}")  # Debugging for OOV tokens
    return torch.tensor(tensor, dtype=torch.long)

class TranslationDataset(Dataset):
    def __init__(self, 
                 src_sentences,
                 tgt_sentences,
                 src_lang_token_to_idx,
                 tgt_lang_token_to_idx
                 ):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_lang_token_to_idx = src_lang_token_to_idx
        self.tgt_lang_token_to_idx = tgt_lang_token_to_idx

    def __len__(self):
        return len(self.src_sentences)
    

    def __getitem__(self, idx):
        src_sent = self.src_sentences[idx]
        tgt_sent = self.tgt_sentences[idx]

        # convert sent to tensors 
        src_tensor = sentence_to_tensor(src_sent, self.src_lang_token_to_idx)
        tgt_tensor = sentence_to_tensor(tgt_sent, self.tgt_lang_token_to_idx)

        return src_tensor, tgt_tensor

# DataLoader collate function to pad sequences in each batch
def collate_fn(batch, src_pad_idx, tgt_pad_idx):
    src_batch, tgt_batch = zip(*batch)

    # Pad sequences 
    src_batch_padded = torch.nn.utils.rnn.pad_sequence(src_batch, padding_value=src_pad_idx, batch_first=True)
    tgt_batch_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, padding_value=tgt_pad_idx, batch_first=True)

    return src_batch_padded, tgt_batch_padded


# save token_2_idx and idx_2_token files 
def save_file(data_dict, file_path):
    with open(file_path, 'w') as f:
        json.dump(data_dict, f)


# main dataset processing pipeline 
def get_data_loaders(train_file, 
                     val_file, 
                     test_file, 
                     batch_size,
                     vocab_dir
                     ):
    train_src, train_tgt = read_csv_data(train_file)
    val_src, val_tgt= read_csv_data(val_file)
    test_src, test_tgt = read_csv_data(test_file)

    # tokenize sequences 
    train_src_tokenized = [tokenize_de(sent) for sent in train_src]
    val_src_tokenized = [tokenize_de(sent) for sent in val_src]
    test_src_tokenized = [tokenize_de(sent) for sent in test_src]

    train_tgt_tokenized = [tokenize_en(sent) for sent in train_tgt]
    val_tgt_tokenized = [tokenize_en(sent) for sent in val_tgt]
    test_tgt_tokenized = [tokenize_en(sent) for sent in test_tgt]

    # build mappings 
    src_token_2_idx, src_idx_2_token = build_vocab(train_src_tokenized, min_freq=1)
    tgt_token_2_idx, tgt_idx_2_token = build_vocab(train_tgt_tokenized, min_freq=1)

    print(f"[INFO] Source vocabulary size: {len(src_token_2_idx)}")
    print(f"[INFO] Target vocabulary size: {len(tgt_token_2_idx)}")

    # Save token_2_idx and idx_2_token mappings to file
    save_file(src_token_2_idx, f'{vocab_dir}/src_token_2_idx.json')
    save_file(tgt_token_2_idx, f'{vocab_dir}/tgt_token_2_idx.json')
    save_file(src_idx_2_token, f'{vocab_dir}/src_idx_2_token.json')
    save_file(tgt_idx_2_token, f'{vocab_dir}/tgt_idx_2_token.json')

    # Create datasets
    train_ds = TranslationDataset(train_src_tokenized, train_tgt_tokenized, src_token_2_idx, tgt_token_2_idx)
    val_ds = TranslationDataset(val_src_tokenized,val_tgt_tokenized,src_token_2_idx, tgt_token_2_idx)
    test_ds = TranslationDataset(test_src_tokenized, test_tgt_tokenized, src_token_2_idx, tgt_token_2_idx)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, src_token_2_idx['<pad>'], tgt_token_2_idx['<pad>']))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, src_token_2_idx['<pad>'], tgt_token_2_idx['<pad>']))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, src_token_2_idx['<pad>'], tgt_token_2_idx['<pad>']))

    return train_dl, val_dl, test_dl, src_token_2_idx, src_idx_2_token, tgt_token_2_idx, tgt_idx_2_token
