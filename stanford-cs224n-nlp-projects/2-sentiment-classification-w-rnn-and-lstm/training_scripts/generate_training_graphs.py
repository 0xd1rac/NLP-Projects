from config.GlobalConfig import *
import os 
import sys 
import json
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def load_logs(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def process_data(data):
    epoch_lis, train_loss_lis, test_loss_lis, test_acc_lis, test_precision_lis, test_recall_lis, test_f1_lis = [], [], [], [], [], [], []
    for unit in data:
        epoch_lis.append(unit['epoch'])
        train_loss_lis.append()
        test_loss_lis.append()
        test_acc_lis.append()
        test_precision_lis.append()
        test_recall_lis.append()
        test_f1_lis.append()

    return epoch_lis, train_loss_lis, test_loss_lis, test_acc_lis, test_precision_lis, test_recall_lis, test_f1_lis

def get_model_to_logs_dict(log_dir):
    log_fnames = os.listdir(GLOBAL_LOG_FILE_DIR)
    model_to_logs = {}
    for fname in log_fnames:
        model = fname.split('.')[0]
        file_path = f"{log_dir}/{fname}"
        data = load_logs(file_path)
        epoch_lis, train_loss_lis, test_loss_lis, test_acc_lis, test_precision_lis, test_recall_lis, test_f1_lis = process_data(data)
        logs = {
            "epochs": epoch_lis,
            "train_loss": train_loss_lis,
            "test_loss": test_loss_lis,
            "test_acc": test_acc_lis,
            "test_precision": test_precision_lis,
            "test_recall": test_recall_lis,
            "test_f1": test_f1_lis
        }
        model_to_logs[model] = logs
    
    return model_to_logs

def plot(model_to_logs_dict, 
         x_key, 
         y_key,
         save_path=None
         ):
    models = list(model_to_logs_dict.keys())

    fig, ax1 = plt.subplots(1, 2, figsize=(14, 6))
    for model in models:
        ax1.plot(model_to_logs_dict[model][x_key], 
                 model_to_logs_dict[model][y_key], 
                 label=f"{model}"
                 )
    ax1.set_title(f"{y_key} over {x_key}")
    ax1.set_xlabel(x_key)
    ax1.set_ylabel(y_key)
    ax1.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_metrics(model_to_logs_dict):
    x_key = "epochs"
    y_key_lis = ["train_loss",
                 "test_loss",
                 "test_acc", 
                 "test_precision",
                 "test_recall",
                 "test_f1"]
    
    for y_key in y_key_lis:
        plot(model_to_logs_dict=model_to_logs_dict,
             x_key=x_key,
             y_key=y_key,
             save_path=f"{GRAPH_DIR}"
             )


if __name__ == "__main__":
    model_to_logs = get_model_to_logs_dict()


    


    

