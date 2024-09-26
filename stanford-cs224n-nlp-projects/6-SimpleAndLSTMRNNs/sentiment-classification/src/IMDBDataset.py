import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
# from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from sklearn.model_selection import train_test_split
import torchtext; torchtext.disable_torchtext_deprecation_warning()

class IMDBDataset(Dataset):
    def __init__(self, data, vocab=None, max_length=256, min_freq=2):
        """
        Custom Dataset class for IMDb movie reviews from a DataFrame.

        Args:
            data (DataFrame): Pandas DataFrame containing the reviews and labels.
            vocab (torchtext.vocab.Vocab, optional): Vocabulary. If not provided, it will be built from the data.
            max_length (int): Maximum length to pad/truncate the reviews.
            min_freq (int): Minimum frequency for including words in the vocabulary.
        """
        self.data = data
        self.max_length = max_length

        # Tokenizer for splitting text into tokens (using basic English tokenizer from TorchText)
        self.tokenizer = get_tokenizer("basic_english")

        # If no vocab is provided, build a new one from the dataset
        if vocab is None:
            self.vocab = self.build_vocab(min_freq=min_freq)
        else:
            self.vocab = vocab

        # Map labels to numerical values (positive -> 1, negative -> 0)
        self.label_map = {"positive": 1, "negative": 0}

    def build_vocab(self, min_freq):
        """
        Build a vocabulary based on the dataset.
        Args:
            min_freq (int): Minimum frequency of words to be included in the vocabulary.
        
        Returns:
            torchtext.vocab.Vocab: The built vocabulary.
        """
        def yield_tokens(data):
            for text in data['review']:
                yield self.tokenizer(text)

        # Build the vocabulary from the dataset
        vocab = build_vocab_from_iterator(yield_tokens(self.data), specials=["<unk>", "<pad>"], min_freq=min_freq)
        vocab.set_default_index(vocab["<unk>"])  # Set default index for unknown tokens

        return vocab

    def pad_sequence(self, sequence):
        """
        Pad or truncate a sequence to a fixed max length.
        Args:
            sequence (list): List of token indices.
        
        Returns:
            torch.Tensor: The padded or truncated sequence.
        """
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
        else:
            sequence += [self.vocab["<pad>"]] * (self.max_length - len(sequence))

        return torch.tensor(sequence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        Args:
            idx (int): Index of the review to get.
        
        Returns:
            tuple: (review tensor, label tensor)
        """
        # Get the review and sentiment from the DataFrame
        text = self.data.iloc[idx]['review']
        label = self.data.iloc[idx]['sentiment']

        # Tokenize the text
        tokens = self.tokenizer(text)

        # Convert tokens to indices
        token_indices = [self.vocab[token] for token in tokens]

        # Pad or truncate the token indices to max_length
        padded_sequence = self.pad_sequence(token_indices)

        # Convert label to a tensor (1 for positive, 0 for negative)
        label_tensor = torch.tensor(self.label_map[label], dtype=torch.long)

        return padded_sequence, label_tensor

# Helper function to create train/test split and DataLoaders
def create_dataloaders(file_path, batch_size=32, test_size=0.2, max_length=256, min_freq=2, random_state=42):
    """
    Create DataLoader for IMDb Dataset with train/test split.

    Args:
        file_path (str): Path to the IMDb dataset file.
        batch_size (int): The number of samples per batch.
        test_size (float): Proportion of the dataset to include in the test split.
        max_length (int): Maximum length of each review.
        min_freq (int): Minimum frequency for including words in the vocabulary.
        random_state (int): Seed for reproducibility of the split.

    Returns:
        tuple: (train_loader, test_loader)
    """
    # Load the dataset into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Perform train/test split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data['sentiment'])

    # Create train and test datasets
    train_dataset = IMDBDataset(data=train_data, max_length=max_length, min_freq=min_freq)
    test_dataset = IMDBDataset(data=test_data, vocab=train_dataset.vocab, max_length=max_length)
    vocab_size = len(train_dataset.vocab)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, vocab_size

# Example usage
# if __name__ == "__main__":
#     # Path to the IMDb CSV file (it should have 'review' and 'sentiment' columns)
#     file_path = "../dataset/IMDB_Dataset.csv"

#     # Create the DataLoaders
#     train_loader, test_loader = create_dataloaders(file_path, 
#                                                    batch_size=32, 
#                                                    test_size=0.2, 
#                                                    max_length=256
#                                                    )

#     # Checking the shape of a batch
#     for batch in train_loader:
#         inputs, labels = batch
#         print("Input batch shape:", inputs.shape)  # (batch_size, max_length)
#         print("Labels batch shape:", labels.shape)  # (batch_size,)
#         break
