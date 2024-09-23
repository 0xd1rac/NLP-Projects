from SimpleRNN import SimpleRNN
from TextDataset import TextDataset
from Trainer import Trainer
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim


EMBEDDING_DIM = 50
HIDDEN_DIM = 128
BATCH_SIZE = 32
SEQ_LEN = 15
DATASET_FILE_NAME = "data.txt"
NUM_EPOCHS = 1

train_ds = TextDataset(DATASET_FILE_NAME , SEQ_LEN)

VOCAB_SIZE = max(train_ds.data) + 1 
# print(len(train_ds.tokens))
# print(len(train_ds.data))



print(f"Max index in dataset: {max(train_ds.data)}")
print(f"Vocab size: {VOCAB_SIZE}")

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
model = SimpleRNN(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

model.to(device)
trainer = Trainer(model, train_dl, criterion, optimizer, device)
trainer.train(NUM_EPOCHS)