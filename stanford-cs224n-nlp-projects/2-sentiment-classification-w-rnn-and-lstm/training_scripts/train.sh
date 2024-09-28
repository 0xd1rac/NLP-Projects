#!/bin/bash

# This script will run all Python training scripts in the `training_scripts` directory.
echo "Starting training process..."

# Sequentially running each training script and printing the filename
echo "Running train_bidirectional_multi_layer_lstm.py"
python3 train_bidirectional_multi_layer_lstm.py

echo "Running train_bidirectional_multi_layer_rnn.py"
python3 train_bidirectional_multi_layer_rnn.py

echo "Running train_bidirectional_single_layer_lstm.py"
python3 train_bidirectional_single_layer_lstm.py

echo "Running train_bidirectional_single_layer_rnn.py"
python3 train_bidirectional_single_layer_rnn.py

echo "Running train_multi_layer_lstm.py"
python3 train_multi_layer_lstm.py

echo "Running train_multi_layer_rnn.py"
python3 train_multi_layer_rnn.py

echo "Running train_single_layer_lstm.py"
python3 train_single_layer_lstm.py

echo "Running train_single_layer_rnn.py"
python3 train_single_layer_rnn.py

echo "Training process completed."
