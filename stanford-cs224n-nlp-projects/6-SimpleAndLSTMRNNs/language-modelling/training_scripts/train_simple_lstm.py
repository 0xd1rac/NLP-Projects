import os 
import sys
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.SimpleLSTM import SimpleLSTM
from src.data.TextDataset import TextDataset
from src.training.Trainer import Trainer
from config.GlobalConfig import *
from config.SimpleLSTMConfig import *

if __name__ == '__main__':
    train_ds = TextDataset(DATASET_FILE_PATH, SEQ_LEN)
    VOCAB_SIZE = max(train_ds.data) + 1 
    train_dl = DataLoader(train_ds,batch_size=BATCH_SIZE, shuffle=True)
    model = SimpleLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    if IS_TEACHER_FORCING:
          print(f"[INFO] Training Simple LSTM for {NUM_EPOCHS} epochs with teacher forcing ratio, {TEACHER_FORCING_RATIO}.")
    else:
          print(f"[INFO] Training Simple LSTM for {NUM_EPOCHS} epochs without teacher forcing.")

    trainer = Trainer(model=model, 
                    train_dl=train_dl, 
                    criterion=criterion, 
                    optimizer=optimizer, 
                    device=device,
                    checkpoint_dir=CHECKPOINT_DIR,
                    log_file=LOG_FILE_PATH,
                    save_interval=CHECKPOINT_INTERVAL,
                    teacher_forcing=IS_TEACHER_FORCING,
                    teacher_forcing_ratio=TEACHER_FORCING_RATIO
                    )
     
    trainer.train(NUM_EPOCHS)