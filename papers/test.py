import torch
from models.gptv2 import GPTv2  # assuming you've defined GPTv2 correctly

# Define constants
VOCAB_SIZE = 1000
D_MODEL = 768
NUM_ENCODER_BLKS = 12
NUM_HEADS = 12
SEQ_LEN = 64
BATCH_SIZE = 32 

# Initialize the GPT model (GPTv2 in your case)
model = GPTv2(vocab_size=VOCAB_SIZE, 
              n_positions=SEQ_LEN,
              n_embd=D_MODEL,
              n_layer=NUM_ENCODER_BLKS,
              n_head=NUM_HEADS)

# Create dummy input data: random token IDs within the range [0, VOCAB_SIZE-1]
dummy_data = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))

# Forward pass: generate predictions
preds = model(dummy_data)

# Output predictions shape (should match batch size, sequence length, and vocab size)
print(preds.shape)  # Output shape: (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
