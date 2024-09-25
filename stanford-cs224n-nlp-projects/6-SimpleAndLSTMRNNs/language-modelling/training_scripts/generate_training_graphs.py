import sys 
import os 
import matplotlib.pyplot as plt
import json

"""
{
    "simple_lstm": 
        {
            "losses": [],
            "perplexity": []
        }
    "simple_rnn":
        {
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
              'multi_layer_lstm.json']

teacher_forcing_logs_fpaths = [f"{TEACHER_FORCING_DIR}/{fname}" for fname in log_fnames]
no_teacher_forcing_logs_fpaths = [f"{NO_TEACHER_FORCING_DIR}/{fname}" for fname in log_fnames]

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


data = load_logs(teacher_forcing_logs_fpaths[0])
process_data(data)




# def plot_logs(teacher_forcing_logs, no_teacher_forcing_logs, metric, title, ylabel):
#     plt.figure(figsize=(10, 6))

#     # Plot teacher forcing logs
#     for i, log in enumerate(teacher_forcing_logs):
#         epochs = [entry['epoch'] for entry in log]
#         values = [entry[metric] for entry in log]
#         plt.plot(epochs, values, label=f'Teacher Forcing {i+1}', linestyle='-', marker='o')

#     # Plot no teacher forcing logs
#     for i, log in enumerate(no_teacher_forcing_logs):
#         epochs = [entry['epoch'] for entry in log]
#         values = [entry[metric] for entry in log]
#         plt.plot(epochs, values, label=f'No Teacher Forcing {i+1}', linestyle='--', marker='x')

#     plt.title(title)
#     plt.xlabel('Epoch')
#     plt.ylabel(ylabel)
#     plt.legend()
#     plt.grid(True)
#     plt.show()



# # Load the logs
# teacher_forcing_logs = [load_logs(fpath) for fpath in teacher_forcing_logs_fpaths]
# no_teacher_forcing_logs = [load_logs(fpath) for fpath in no_teacher_forcing_logs_fpaths]

# plot_logs(teacher_forcing_logs, no_teacher_forcing_logs, 'loss', 'Training Loss Over Epochs', 'Loss')

# plot_logs(teacher_forcing_logs, no_teacher_forcing_logs, 'perplexity', 'Perplexity Over Epochs', 'Perplexity')
