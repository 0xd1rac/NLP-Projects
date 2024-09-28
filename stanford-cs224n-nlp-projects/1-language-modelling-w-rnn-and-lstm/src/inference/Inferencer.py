import torch.nn as nn
import torch.nn.functional as F
import torch

class Inferencer:
    def __init__(self,
                 model: nn.Module,
                 idx_to_token: dict,
                 token_to_idx: dict,
                 device: torch.device
                 ):
        
        """
        Inferencer class to handle text generation with top-k decoding.

        Args:
            model (nn.Module): The trained language model.
            idx_to_token (dict): Dictionary mapping token indices to words.
            token_to_idx (dict): Dictionary mapping words to token indices.
            device (torch.device): The device (CPU or GPU) to run the inference on.
        """

        self.model = model
        self.model.eval()
        self.idx_to_token = idx_to_token
        self.token_to_idx = token_to_idx
        self.device = device
    
    def top_k_sample(self,
                     logits: torch.Tensor,
                     k: int
                     ) -> int:
        """
        Top-k sampling from the logits.

        Args:
            logits (torch.Tensor): The logits from the model's output.
            k (int): The number of top tokens to consider for sampling.

        Returns:
            next_token (int): The sampled token index from the top-k.
        """

        top_k_logits, top_k_indices = torch.topk(logits, k)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        next_token = torch.multinomial(top_k_probs, num_samples=1).item()
        next_token = top_k_indices[next_token].item()
        return next_token

    def generate_text(self,
                      start_sequence: torch.Tensor,
                      max_len: int,
                      k:int=5,
                      temperature: float=1.0
                      ):
        """
        Generate text using the language model with top-k decoding.

        Args:
            start_sequence (torch.Tensor): The starting input sequence (token indices).
            max_len (int): The maximum number of tokens to generate.
            k (int): The top-k value for sampling.
            temperature (float): Softmax temperature to control the randomness of predictions.

        Returns:
            generated_text (str): The generated text.
        """

        generated_tokens = start_sequence.tolist()
        input_seq = start_sequence.unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(max_len):
                output = self.model(input_seq)
                output = output.squeeze(0)
                logits = output / temperature
                next_token = self.top_k_sample(logits, k)

                generated_tokens.append(next_token)
                input_seq = torch.tensor([next_token], device=self.device).unsqueeze(0)
        
        generated_text = ' '.join([self.idx_to_token[token] for token in generated_tokens])

        return generated_text