import torch
import torch.nn as nn
from tqdm import tqdm
import os
import json
import nltk
from nltk.translate.bleu_score import sentence_bleu

class Seq2SeqTrainer:
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
                 src_pad_idx,
                 trg_pad_idx,
                 trg_vocab,
                 clip_grad=1.0,
                 teacher_forcing_ratio=0.5,
                 is_preload=False,     
                 preload_path=None):
        
        self.model = model
        self.train_dl = train_dl
        self.test_dl = test_dl  
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device 
        self.checkpoint_dir = checkpoint_dir
        self.log_file = log_file
        self.save_interval = save_interval
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.clip_grad = clip_grad
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_vocab = trg_vocab  # Needed for decoding indices to words
        
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
            print(f"Loaded checkpoint '{preload_path}' (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            print(f"No checkpoint found at '{preload_path}'")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dl)
        progress_bar = tqdm(self.train_dl, desc=f"Epoch {epoch}", leave=False)

        for batch_idx, (src, trg) in enumerate(progress_bar):
            src, trg = src.to(self.device), trg.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            output = self.model(src, trg, teacher_forcing_ratio=self.teacher_forcing_ratio)
            # trg shape: [batch_size, trg_len]
            # output shape: [batch_size, trg_len, output_dim]

            # Shift target to ignore the first token (<sos> token) for loss calculation
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)  # remove <sos> and reshape for loss
            trg = trg[:, 1:].reshape(-1)  # remove <sos> and flatten

            # Calculate loss
            loss = self.criterion(output, trg)

            total_loss += loss.item()

            # Backpropagation and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

        avg_loss = total_loss / num_batches
        progress_bar.set_postfix(loss=avg_loss)
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        total_bleu = 0
        num_batches = len(self.test_dl)
        
        with torch.no_grad():
            for src, trg in tqdm(self.test_dl, desc="Evaluating", leave=False):
                src, trg = src.to(self.device), trg.to(self.device)

                output = self.model(src, trg, 0)  # turn off teacher forcing for evaluation
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)

                loss = self.criterion(output, trg)
                total_loss += loss.item()

                # Convert output predictions to tokens and calculate BLEU score
                predicted_tokens = self.decode_batch(output)
                reference_tokens = self.decode_batch(trg, is_reference=True)
                bleu_score = self.calculate_bleu(predicted_tokens, reference_tokens)
                total_bleu += bleu_score

        avg_loss = total_loss / num_batches
        avg_bleu = total_bleu / num_batches
        return avg_loss, avg_bleu

    def decode_batch(self, batch, is_reference=False):
        """
        Converts a batch of word indices into human-readable token sequences.
        """
        decoded_sentences = []
        for i in range(batch.shape[0]):  # iterate through each sentence in batch
            sentence = []
            for idx in batch[i]:
                if is_reference:
                    word = self.trg_vocab.get_itos()[idx]  # Reference decoding (no <unk>)
                else:
                    word = self.trg_vocab.get_itos()[idx.item()]  # Prediction decoding
                if word == "<eos>":  # End of sequence token
                    break
                sentence.append(word)
            decoded_sentences.append(sentence)
        return decoded_sentences

    def calculate_bleu(self, predicted_sentences, reference_sentences):
        """
        Computes BLEU score for a batch of predicted and reference sentences.
        """
        bleu_scores = []
        for pred, ref in zip(predicted_sentences, reference_sentences):
            bleu_scores.append(sentence_bleu([ref], pred, weights=(0.25, 0.25, 0.25, 0.25)))
        return sum(bleu_scores) / len(bleu_scores)

    def save_checkpoint(self, epoch, current_loss, current_bleu):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")

        print(f"Checkpoint saved for epoch {epoch} with loss {current_loss:.4f} and BLEU score {current_bleu:.4f}.")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_loss': current_loss,  # Save the current loss at this point
            'current_bleu': current_bleu  # Save the current BLEU score at this point
        }
        torch.save(checkpoint, checkpoint_path)

    def log_metrics(self, epoch, train_loss, test_loss, bleu_score):
        epoch_metrics = {
            'epoch': epoch, 
            'train_loss': train_loss, 
            'test_loss': test_loss,
            'bleu_score': bleu_score
        }
        
        with open(self.log_file, 'r+') as f:
            logs = json.load(f)
            logs.append(epoch_metrics)
            f.seek(0)
            json.dump(logs, f, indent=4)

    def train(self, num_epochs):
        print(f"Training using {self.device}")
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            print(f"Epoch {epoch} completed. Train Loss: {train_loss:.4f}")

            test_loss, bleu_score = self.evaluate()
            print(f"Test Loss: {test_loss:.4f}, BLEU Score: {bleu_score:.4f}")

            self.log_metrics(epoch, train_loss, test_loss, bleu_score)

            # Save checkpoint after every evaluation with current loss and BLEU score
            self.save_checkpoint(epoch, test_loss, bleu_score)

            if epoch % self.save_interval == 0:
                self.save_checkpoint(epoch, test_loss, bleu_score)
