import sys 
import os 
import matplotlib.pyplot as plt
import json
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.Inferencer import Inferencer
from src.models.MultiLayerRNN import MultiLayerRNN
from config.GlobalConfig import *
from config.MultiLayerRNNConfig import *
from src.data.TextDataset import TextDataset


if __name__ == '__main__':
    train_ds = TextDataset(DATASET_FILE_PATH, SEQ_LEN)

    VOCAB_SIZE = max(train_ds.data) + 1 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    model = MultiLayerRNN(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS
        )
    model.to(device)
    checkpoint_path = "../outputs/model_checkpoints/teacher_forcing/multi_layer_rnn/best_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(len(train_ds.idx_to_token.keys()))
    inferencer = Inferencer(model, 
                            train_ds.idx_to_token, 
                            train_ds.token_2_idx, 
                            device=device
                            )
    
    # print(train_ds.token_2_idx.keys())
    # Sample start sequence, e.g., "[SOS]" token to begin generation
    start_sequence = torch.tensor([train_ds.token_2_idx["solemnly"]]).to(torch.device('cpu'))
    # Generate text using the model with top-k sampling
    generated_text = inferencer.generate_text(start_sequence, 
                                              max_len=500, 
                                              k=50, 
                                              temperature=1.0)

    print(f"Generated Text: {generated_text}")
