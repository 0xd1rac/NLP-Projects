import torch
import torch.nn
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
                 save_interval,
                 teacher_forcing=False,
                 teacher_forcing_ratio=0.5
                 ):
        self.model = model
        self.train_dl = train_dl
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device 
        self.checkpoint_dir = checkpoint_dir
        self.best_loss = float('inf')
        self.log_file = log_file
        self.save_interval = save_interval
        self.teacher_forcing = teacher_forcing
        self.teacher_forcing_ratio = teacher_forcing_ratio

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        log_file_dir = os.path.dirname(self.log_file)
        os.makedirs(log_file_dir, exist_ok=True)

        with open(self.log_file, 'w') as f:
            json.dump([], f)

    # def train_one_epoch(self, epoch):
    #     self.model.train()
    #     total_loss = 0
    #     num_batches = len(self.train_dl)
    #     progress_bar = tqdm(self.train_dl, desc=f"Epoch {epoch}", leave=False)
    #     for batch_idx, (inputs, targets) in enumerate(progress_bar):
    #         inputs, targets = inputs.to(self.device), targets.to(self.device)

    #         self.optimizer.zero_grad()

    #         outputs = self.model(inputs)
    #         print(f"outputs shape: {outputs.shape}")
           
    #         if self.teacher_forcing and torch.rand(1).item() < self.teacher_forcing_ratio:
    #             targets_for_loss = targets[:, 1:] # Use the actual previous target for teacher forcing
    #             targets_for_loss = targets_for_loss.float()
    #             print(f"targets_for_loss shape: {targets_for_loss.shape}")
    #         else: 
    #             targets_for_loss = outputs

    #         targets = targets[:,-1]

    #         loss = self.criterion(targets_for_loss, targets)
    #         total_loss += loss.item()

    #         loss.backward()
    #         clip_grad_norm_(self.model.parameters(), max_norm=1.0)

    #         self.optimizer.step()

    #     avg_loss = total_loss / (batch_idx + 1)
    #     perplexity = math.exp(avg_loss) if avg_loss < 300 else float('inf')  # Safeguard for overflow
    #     progress_bar.set_postfix(loss=avg_loss, perplexity=perplexity)

    #     return avg_loss, perplexity
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dl)
        progress_bar = tqdm(self.train_dl, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size, sequence_length = inputs.shape  # Assuming inputs is [batch_size, sequence_length]

            self.optimizer.zero_grad()

            # Initialize the first input sequence to the model
            input_seq = inputs[:, 0].unsqueeze(1)  # Start with the first token, shape [batch_size, 1]
            outputs = []  # To store all outputs over the sequence

            # Iterate over the sequence (from 1 onwards, as input_seq starts with token 0)
            for t in range(1, sequence_length):
                output = self.model(input_seq)  # Forward pass for the current input
                outputs.append(output.squeeze(1))  # Collect the output, shape [batch_size, vocab_size]

                # Determine the next input based on teacher forcing
                if self.teacher_forcing and torch.rand(1).item() < self.teacher_forcing_ratio:
                    # Use the ground truth target as the next input
                    input_seq = targets[:, t].unsqueeze(1)  # Use actual target, shape [batch_size, 1]
                else:
                    # Use the model's own prediction as the next input
                    input_seq = output.argmax(dim=-1).unsqueeze(1)  # Use model prediction, shape [batch_size, 1]

            outputs = torch.stack(outputs, dim=1)  # Stack along time dimension, shape [batch_size, sequence_length-1, vocab_size]

            # Compute the loss: compare the model's predictions with the actual targets
            targets_for_loss = targets[:, 1:].contiguous().view(-1)  # Flatten targets, shape [batch_size * (sequence_length - 1)]
            outputs = outputs.view(-1, outputs.size(-1))  # Flatten outputs, shape [batch_size * (sequence_length - 1), vocab_size]

            loss = self.criterion(outputs, targets_for_loss)  # Compute cross-entropy loss
            total_loss += loss.item()

            # Backpropagation and optimization step
            loss.backward()
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss) if avg_loss < 300 else float('inf')  # Safeguard for overflow
        progress_bar.set_postfix(loss=avg_loss, perplexity=perplexity)

        return avg_loss, perplexity


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

    def log_metrics(self, epoch, avg_loss, perplexity):
        epoch_metrics = {'epoch': epoch, 'loss': avg_loss, 'perplexity': perplexity}
        
        with open(self.log_file, 'r+') as f:
            logs = json.load(f)
            logs.append(epoch_metrics)
            f.seek(0)
            json.dump(logs, f, indent=4)

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            avg_loss, perplexity = self.train_one_epoch(epoch)
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")

            self.log_metrics(epoch, avg_loss, perplexity)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_checkpoint(epoch, is_best=True)
            
            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch)