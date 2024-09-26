import sys 
import os 
import torchtext; torchtext.disable_torchtext_deprecation_warning()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.SentimentRNN import SentimentRNN
from src.IMDBDataset import *
import json
from config.GlobalConfig import *
# from config.RNNConfig import *

if __name__ == '__main__':
    train_dl, test_dl, vocab_size = create_dataloaders(file_path="../dataset/IMDB_Dataset.csv")
    sing_lay_rnn = SentimentRNN(vocab_size=vocab_size,
                                embedding_dim=EMBEDDING_DIM,
                                hidden_dim=HIDDEN_DIM,
                                num_layers=1,
                                is_bidirectional=False,
                                output_size=1,
                                dropout=0.5
                                )
    # bi_dir_sing_lay_rnn

    # mult_lay_rnn
    # bi_dir_mult_lay_rnn


