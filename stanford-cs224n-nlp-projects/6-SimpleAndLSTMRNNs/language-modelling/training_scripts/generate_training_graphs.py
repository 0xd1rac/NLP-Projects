import sys 
import os 
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.GlobalConfig import *

"""
{
    "simple_lstm": 
        {
            "epochs": []
            "losses": [],
            "perplexity": []
        }
    "simple_rnn":
        {
            "epochs": []
            "losses": [],
            "perplexity": []
        }   
}

"""

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.GlobalConfig import *
TEACHER_FORCING_DIR = f"{GLOBAL_LOG_FILE_DIR}/teacher_forcing"
NO_TEACHER_FORCING_DIR = f"{GLOBAL_LOG_FILE_DIR}/no_teacher_forcing"

log_fnames = ['simple_rnn.json',
              'simple_lstm.json',
              'multi_layer_rnn.json', 
              'multi_layer_lstm.json'
              ]


# Load the log data from the JSON files
def load_logs(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_data(data):
    epoch_lis, loss_lis, perplexity_lis = [], [], []
    for unit in data:
        epoch_lis.append(unit['epoch'])
        loss_lis.append(unit['loss'])
        perplexity_lis.append(unit['perplexity'])
    return epoch_lis, loss_lis, perplexity_lis

def get_model_to_logs_dict(log_dir, log_fnames):
    model_to_logs = {}
    for fname in log_fnames:
        model = fname.split('.')[0]
        file_path = f"{log_dir}/{fname}"
        data = load_logs(file_path)
        epoch_lis, loss_lis, perplexity_lis = process_data(data)
        logs = {
            'epochs': epoch_lis,
            'losses': loss_lis,
            "perplexity": perplexity_lis
        }
        model_to_logs[model] = logs
    
    return model_to_logs

import matplotlib.pyplot as plt

def plot_combined_metrics(data_w_tf, data_wo_tf, save_path=None):
    # Extracting models from data
    models_w_tf = list(data_w_tf.keys())
    models_wo_tf = list(data_wo_tf.keys())
    
    # Prepare the plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot losses
    for model in models_w_tf:
        ax1.plot(data_w_tf[model]["epochs"], data_w_tf[model]["losses"], label=f"{model} (With TF)")
    for model in models_wo_tf:
        ax1.plot(data_wo_tf[model]["epochs"], data_wo_tf[model]["losses"], label=f"{model} (Without TF)", linestyle='--')
    
    ax1.set_title('Losses over Epochs')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Plot perplexity
    for model in models_w_tf:
        ax2.plot(data_w_tf[model]["epochs"], data_w_tf[model]["perplexity"], label=f"{model} (With TF)")
    for model in models_wo_tf:
        ax2.plot(data_wo_tf[model]["epochs"], data_wo_tf[model]["perplexity"], label=f"{model} (Without TF)", linestyle='--')
    
    ax2.set_title('Perplexity over Epochs')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Perplexity')
    ax2.legend()

    # Show the plots
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plots saved to {save_path}")
    else:
        plt.show()

# Example usage
model_to_logs_w_tf = get_model_to_logs_dict(log_dir=TEACHER_FORCING_DIR, log_fnames=log_fnames)
model_to_logs_wo_tf = get_model_to_logs_dict(log_dir=NO_TEACHER_FORCING_DIR, log_fnames=log_fnames)

plot_combined_metrics(model_to_logs_w_tf, model_to_logs_wo_tf, save_path=GRAPH_DIR)
