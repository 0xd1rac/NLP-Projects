from SentimentLSTM import SentimentLSTM
from SentimentRNN import SentimentRNN
import torch 

if __name__ == "__main__":
    # Example Usage
    vocab_size = 10000  # Example vocab size
    embedding_dim = 300  # Example embedding dimension
    hidden_dim = 256  # Number of hidden units in LSTM
    num_layers = 2  # Number of LSTM layers
    is_bidirectional = True  # Bidirectional LSTM
    output_size = 1  # Output size (binary sentiment classification)

    # Instantiate the model
    lstm_model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim, num_layers, is_bidirectional, output_size)
    rnn_model = SentimentRNN(vocab_size, embedding_dim, hidden_dim, num_layers, is_bidirectional, output_size)
    # Example input: a batch of sentences with word indices (batch_size=32, sequence_length=50)
    input_data = torch.randint(0, vocab_size, (32, 50))  # Random token indices
    lstm_output = lstm_model(input_data)
    rnn_output = rnn_model(input_data)
    print(rnn_output.shape)
    print(lstm_output.shape)  # Should output (batch_size, output_size)