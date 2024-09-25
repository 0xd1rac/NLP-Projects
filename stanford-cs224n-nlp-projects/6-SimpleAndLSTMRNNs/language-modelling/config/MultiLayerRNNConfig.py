from .GlobalConfig import *

NUM_EPOCHS=100
NUM_LAYERS=4
IS_TEACHER_FORCING = True
TEACHER_FORCING_RATIO=0.8

if IS_TEACHER_FORCING:
    CHECKPOINT_DIR=f"{GLOBAL_CHECKPOINT_DIR}/teacher_forcing/multi_layer_rnn"
    LOG_FILE_PATH=f"{GLOBAL_LOG_FILE_DIR}/teacher_forcing/multi_layer_rnn.json"
else:
    CHECKPOINT_DIR=f"{GLOBAL_CHECKPOINT_DIR}/no_teacher_forcing/multi_layer_rnn"
    LOG_FILE_PATH=f"{GLOBAL_LOG_FILE_DIR}/no_teacher_forcing/multi_layer_rnn.json"

