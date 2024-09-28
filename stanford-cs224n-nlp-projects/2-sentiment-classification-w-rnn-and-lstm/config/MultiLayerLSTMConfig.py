from .GlobalConfig import *

NUM_EPOCHS=10
FILE_NAME="multi_layer_lstm"
CHECKPOINT_DIR=f"{GLOBAL_CHECKPOINT_DIR}/{FILE_NAME}"
LOG_FILE_PATH=f"{GLOBAL_LOG_FILE_DIR}/{FILE_NAME}.json"
PRELOADS=True
PRELOADS_CHECKPOINT_PATH=f"{GLOBAL_CHECKPOINT_DIR}/{FILE_NAME}/best_checkpoint.pth"
NUM_LAYERS=10
