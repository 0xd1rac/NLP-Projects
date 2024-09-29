import torch
import torch.nn as nn
from tqdm import tqdm
import math
import os
from torch.nn.utils import clip_grad_norm_
import json
from sklearn.metrics import precision_score, recall_score, f1_score

class Trainer:
    def __init__(self,
                 model,
                 train_dl,
                 test_dl,   
                 criterion,
                 optimizer,
                 device,
                 checkpoint_dir,
                 log_file,
                 save_interval,
                 is_preload=False,     
                 preload_path=None):   
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl  
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

        # Preload model and optimizer state if is_preload is True and preload_path is provided
        if is_preload and preload_path:
            self.load_checkpoint(preload_path)

    def load_checkpoint(self, preload_path):
        """Loads the model and optimizer states from a checkpoint."""
        if os.path.isfile(preload_path):
            print(f"Loading checkpoint from '{preload_path}'...")
            checkpoint = torch.load(preload_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_loss = checkpoint.get('best_loss', float('inf'))
            print(f"Loaded checkpoint '{preload_path}' (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            print(f"No checkpoint found at '{preload_path}'")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0  
        total_samples = 0
        num_batches = len(self.train_dl)
        progress_bar = tqdm(self.train_dl, desc=f"Epoch {epoch}", leave=False)

        all_targets = []
        all_predictions = []

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)  
            loss = self.criterion(outputs, targets)

            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Calculate accuracy and store predictions/targets for metrics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        avg_loss = total_loss / num_batches
        accuracy = correct / total_samples

        # Calculate F1, precision, recall
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        progress_bar.set_postfix(loss=avg_loss, accuracy=accuracy)

        return avg_loss, accuracy, precision, recall, f1

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total_samples = 0
        num_batches = len(self.test_dl)

        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for inputs, targets in tqdm(self.test_dl, desc="Evaluating", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Calculate accuracy and store predictions/targets for metrics
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        avg_loss = total_loss / num_batches
        accuracy = correct / total_samples

        # Calculate F1, precision, recall
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        return avg_loss, accuracy, precision, recall, f1

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

    def log_metrics(self, epoch, train_loss, train_accuracy, test_loss, test_accuracy, train_precision, train_recall, train_f1, test_precision, test_recall, test_f1):
        epoch_metrics = {
            'epoch': epoch, 
            'train_loss': train_loss, 
            'train_accuracy': train_accuracy,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'test_loss': test_loss, 
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        }
        
        with open(self.log_file, 'r+') as f:
            logs = json.load(f)
            logs.append(epoch_metrics)
            f.seek(0)
            json.dump(logs, f, indent=4)

    def train(self, num_epochs):
        print(f"Training using {self.device}")
        for epoch in range(1, num_epochs + 1):
            train_loss, train_accuracy, train_precision, train_recall, train_f1 = self.train_one_epoch(epoch)
            print(f"Epoch {epoch} completed. Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}, Train F1: {train_f1:.4f}")

            test_loss, test_accuracy, test_precision, test_recall, test_f1 = self.evaluate()
            print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")

            self.log_metrics(epoch, train_loss, train_accuracy, test_loss, test_accuracy, train_precision, train_recall, train_f1, test_precision, test_recall, test_f1)

            if test_loss < self.best_loss:  
                self.best_loss = test_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)
