import torch
import torch.nn as nn
from tqdm import tqdm
import math
import os
from torch.nn.utils import clip_grad_norm_
import json

class Trainer:
    def __init__(self,
                 model,
                 train_dl,
                 criterion,
                 optimizer,
                 device,
                 checkpoint_dir,
                 log_file,
                 save_interval):
        self.model = model
        self.train_dl = train_dl
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device 
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')
        self.log_file = log_file
        self.save_interval = save_interval
        model.to(device)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        log_file_dir = os.path.dirname(self.log_file)
        os.makedirs(log_file_dir, exist_ok=True)

        with open(self.log_file, 'w') as f:
            json.dump([], f)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0  # Track the number of correct predictions
        total_samples = 0
        num_batches = len(self.train_dl)
        progress_bar = tqdm(self.train_dl, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)  # Outputs should be [batch_size, num_classes]
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / num_batches
        accuracy = correct / total_samples

        progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

        return avg_loss, accuracy

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
            print(f"New best model saved with loss {self.best_loss:.4f} at epoch {epoch}.")
        else:
            print(f"Checkpoint saved for epoch {epoch}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(checkpoint, checkpoint_path)

    def log_metrics(self, epoch, avg_loss, accuracy):
        epoch_metrics = {'epoch': epoch, 'loss': avg_loss, 'accuracy': accuracy}
        
        with open(self.log_file, 'r+') as f:
            logs = json.load(f)
            logs.append(epoch_metrics)
            f.seek(0)
            json.dump(logs, f, indent=4)

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            avg_loss, accuracy = self.train_one_epoch(epoch)
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            self.log_metrics(epoch, avg_loss, accuracy)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)
