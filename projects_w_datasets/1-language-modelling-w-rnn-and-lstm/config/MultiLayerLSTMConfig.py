from .GlobalConfig import *

NUM_EPOCHS=100
IS_TEACHER_FORCING = True
TEACHER_FORCING_RATIO=0.8
NUM_LAYERS=4
if IS_TEACHER_FORCING:
    CHECKPOINT_DIR=f"{GLOBAL_CHECKPOINT_DIR}/teacher_forcing/multi_layer_lstm"
    LOG_FILE_PATH=f"{GLOBAL_LOG_FILE_DIR}/teacher_forcing/multi_layer_lstm.json"
else:
    CHECKPOINT_DIR=f"{GLOBAL_CHECKPOINT_DIR}/no_teacher_forcing/multi_layer_lstm"
    LOG_FILE_PATH=f"{GLOBAL_LOG_FILE_DIR}/no_teacher_forcing/multi_layer_lstm.json"

