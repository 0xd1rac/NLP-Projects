import torch
import torch.nn
from tqdm import tqdm
import math

class Trainer:
    def __init__(self,
                 model,
                 train_dl,
                 criterion,
                 optimizer,
                 device
                 ):
        self.model = model
        self.train_dl = train_dl
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device 

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dl)
        progress_bar = tqdm(self.train_dl, desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            print(f"Output shape: {outputs.shape}")  # Should be (batch_size, vocab_size)
            print(f"Target shape: {targets.shape}")  # Should be (batch_size,)

            targets = targets[:,-1]

            loss = self.criterion(outputs, targets)
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()

        avg_loss = total_loss / (batch_idx + 1)
        perplexity = math.exp(avg_loss) if avg_loss < 300 else float('inf')  # Safeguard for overflow
        progress_bar.set_postfix(loss=avg_loss, perplexity=perplexity)

        return avg_loss, perplexity

    def train(self, num_epochs):
        for epoch in range(1, num_epochs + 1):
            avg_loss, perplexity = self.train_one_epoch(epoch)
            print(f"Epoch {epoch} completed. Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
