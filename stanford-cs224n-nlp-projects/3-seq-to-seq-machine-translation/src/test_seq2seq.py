from EncoderBiDirGRU import EncoderBiDirGRU 
from DecoderGRU import DecoderGRU
from Seq2Seq import Seq2Seq
import torch

if __name__ == "__main__":
    EMBEDDING_DIM = 50
    SRC_VOCAB_SIZE = 1000
    TGT_VOCAB_SIZE = 1000
    HIDDEN_DIM = 128
    NUM_LAYERS = 3
    DROPOUT = 0.5
    BATCH_SIZE = 16
    SRC_SEQ_LEN = 20
    START_TOKEN_IDX = 1  # Assuming 1 is the index for <start> token
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = EncoderBiDirGRU(
        SRC_VOCAB_SIZE,
        EMBEDDING_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        DROPOUT
    )
    
    decoder = DecoderGRU(
        target_vocab_size=TGT_VOCAB_SIZE,
        embed_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )

    seq2seq_net = Seq2Seq(
        encoder = encoder,
        decoder = decoder,
        start_token_idx=1,
        eos_token_idx=2,
        device=DEVICE,
        use_attention=True
    )

    src_dummy_data = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN))
    tgt_dummy_data = torch.randint(0, SRC_VOCAB_SIZE, (BATCH_SIZE, SRC_SEQ_LEN))

    output_train = seq2seq_net(src_seq=src_dummy_data,
                               tgt_seq=tgt_dummy_data,
                               teacher_forcing_ratio=0.5
                               )
    
    output_inference = seq2seq_net(src_seq=src_dummy_data,
                                   max_len=10
                                   )
    print(f'Train output: {output_train.shape}')
    print(f'Inference output: {output_inference.shape}')



