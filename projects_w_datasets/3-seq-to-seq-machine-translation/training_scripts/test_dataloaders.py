import os 
import sys
import torch.optim as optim
import torch 
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.TranslationDataset import *
from src.Seq2Seq import Seq2Seq
from src.Seq2SeqTrainer import Seq2SeqTrainer
from src.EncoderBiDirGRU import EncoderBiDirGRU
from src.DecoderGRU import DecoderGRU

if __name__ == "__main__":
    TRAIN_DATASET_PATH = "../dataset/val.csv"
    VAL_DATASET_PATH = "../dataset/val.csv"
    TEST_DATASET_PATH = "../dataset/test.csv"
    VOCAB_DIR = "../dataset/"
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    SAVE_INTERVAL = 5
    GLOBAL_CHECKPOINT_DIR = "../outputs/model_checkpoints"
    LOG_FILE = './training_logs.json'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dl, val_dl, test_dl, src_token_2_idx, src_idx_2_token, tgt_token_2_idx, tgt_idx_2_token = get_data_loaders(
        train_file=TRAIN_DATASET_PATH, 
        val_file=VAL_DATASET_PATH, 
        test_file=TEST_DATASET_PATH, 
        batch_size=BATCH_SIZE,
        vocab_dir=VOCAB_DIR
    )

    SRC_VOCAB_SIZE = len(src_token_2_idx)
    TGT_VOCAB_SIZE = len(tgt_token_2_idx)
    EMBEDDING_DIM = 256
    HIDDEN_DIM = 512
    NUM_LAYERS = 3
    DROPOUT = 0.5

    encoder = EncoderBiDirGRU(
        SRC_VOCAB_SIZE,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT
    )
    
    decoder = DecoderGRU(
        target_vocab_size=TGT_VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    seq2seq_net = Seq2Seq(
        encoder = encoder,
        decoder = decoder,
        start_token_idx=1,
        eos_token_idx=2,
        device=DEVICE,
        use_attention=True
    )
    tgt_pad_idx = tgt_token_2_idx['<pad>']
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)
    optimizer = optim.Adam(seq2seq_net.parameters())


    trainer = Seq2SeqTrainer(
        model=seq2seq_net,
        train_dl=train_dl,
        test_dl=test_dl,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
        checkpoint_dir=GLOBAL_CHECKPOINT_DIR,
        log_file=LOG_FILE,
        save_interval=SAVE_INTERVAL,
        src_token_2_idx=src_token_2_idx,
        tgt_token_2_idx=tgt_token_2_idx,
        tgt_idx_2_token=tgt_idx_2_token,
        teacher_forcing_ratio=0.5
    )
    
    trainer.train(NUM_EPOCHS)





