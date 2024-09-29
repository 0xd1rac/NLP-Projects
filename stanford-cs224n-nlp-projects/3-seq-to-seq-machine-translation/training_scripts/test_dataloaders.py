import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.TranslationDataset import get_data_loaders

if __name__ == "__main__":
    TRAIN_DATASET_PATH = "../dataset/val.csv"
    VAL_DATASET_PATH = "../dataset/val.csv"
    TEST_DATASET_PATH = "../dataset/test.csv"
    VOCAB_DIR = "../dataset/"
    BATCH_SIZE = 32
    train_dl, val_dl, test_dl, src_token_2_idx, tgt_token_2_idx = get_data_loaders(
        train_file=TRAIN_DATASET_PATH, 
        val_file=VAL_DATASET_PATH, 
        test_file=TEST_DATASET_PATH, 
        batch_size=BATCH_SIZE,
        vocab_dir=VOCAB_DIR
    )

    for batch in train_dl:
        src_batch, trg_batch = batch
        print("Source batch shape:", src_batch.shape)
        print("Target batch shape:", trg_batch.shape)
        

