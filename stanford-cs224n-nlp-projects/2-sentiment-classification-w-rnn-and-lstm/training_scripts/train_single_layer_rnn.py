import sys 
import os 
import torchtext; torchtext.disable_torchtext_deprecation_warning()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.SentimentRNN import SentimentRNN
from src.IMDBDataset import *
import json
from config.GlobalConfig import *
import torch.optim as optim
import torch.nn as nn
from src.Trainer import Trainer
from config.SingleLayerRNNConfig import *

if __name__ == '__main__':
    train_dl, test_dl, vocab_size = create_dataloaders(file_path=DATASET_FILE_PATH)
    model = SentimentRNN(vocab_size=vocab_size,
                                embedding_dim=EMBEDDING_DIM,
                                hidden_dim=HIDDEN_DIM,
                                num_layers=1,
                                is_bidirectional=False,
                                output_size=2,
                                dropout=0.5
                                )
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    trainer = Trainer(model=model,
                      train_dl=train_dl,
                      test_dl=test_dl,
                      criterion=criterion, 
                      optimizer=optimizer, 
                      device=device,
                      checkpoint_dir=CHECKPOINT_DIR,
                      log_file=LOG_FILE_PATH,
                      save_interval=CHECKPOINT_INTERVAL,
                      is_preload=IS_PRELOAD,
                      preload_path=PRELOAD_CHECKPOINT_PATH
                      )

    trainer.train(NUM_EPOCHS)



