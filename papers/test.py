import torch
from models.transformer import Transformer  # assuming you've defined GPTv2 correctly

# Define constants
SRC_VOCAB_SIZE = 1000
TGT_VOCAB_SIZE = 1000
D_MODEL = 768
NUM_LAYERS = 6
NUM_HEADS = 12
D_FF = 2048
SEQ_LEN = 64
BATCH_SIZE = 32
MAX_SEQ_LEN = 100  # Maximum sequence length for positional embeddings

# Initialize the Transformer model
model = Transformer(
    src_vocab_size=SRC_VOCAB_SIZE, 
    tgt_vocab_size=TGT_VOCAB_SIZE, 
    n_embd=D_MODEL,
    n_layer=NUM_LAYERS,
    n_head=NUM_HEADS,
    d_ff=D_FF,
    max_seq_len=MAX_SEQ_LEN
)

# Create dummy input data: random token IDs within the range [0, SRC_VOCAB_SIZE-1] and [0, TGT_VOCAB_SIZE-1]
src_dummy_data = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))  # Source input data
tgt_dummy_data = torch.randint(0, TGT_VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN))  # Target input data

# Forward pass: generate predictions
preds = model(src_dummy_data, tgt_dummy_data)

# Output predictions shape (should match batch size, sequence length, and target vocabulary size)
print(preds.shape)  # Output shape: (BATCH_SIZE, SEQ_LEN, TGT_VOCAB_SIZE)