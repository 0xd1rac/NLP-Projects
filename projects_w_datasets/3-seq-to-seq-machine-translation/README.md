## Overview

The Seq2Seq models are designed to perform sequence-to-sequence tasks such as machine translation, where an input sequence (e.g., a sentence in a source language) is encoded into a fixed representation and then decoded to produce an output sequence (e.g., a sentence in a target language).

In this project:
- **Seq2Seq without Attention**: The decoder only uses the final hidden state from the encoder to generate the output sequence.
- **Seq2Seq with Attention**: The decoder uses a dynamic context vector from the encoder hidden states (computed via the attention mechanism) at each decoding timestep.

## Dataset

**Multi30k (EN-DE Translation)**:
   - **Description**: A dataset for English-German translation that contains around 30,000 parallel sentences. It’s much smaller compared to larger translation datasets like WMT, making it more suitable for training on limited hardware.
   - **Size**: ~30k sentence pairs.
   - **Task**: Machine translation.
   - **Link**: [Multi30k on Kaggle](https://www.kaggle.com/datasets/mohamedlotfy50/wmt-2014-english-german)

## Model Architecture

### 1. Encoder (Bidirectional GRU)
- **Type**: 3-layer bidirectional GRU.
- **Input**: A sequence of token embeddings (e.g., words or subwords).
- **Output**: A sequence of hidden states for each timestep, and a final hidden state for both the forward and backward passes.

### 2. Decoder (GRU)
- **Type**: 3-layer unidirectional GRU.
- **Input**: The previous token's embedding and, optionally, the attention-weighted context vector (if attention is used).
- **Output**: A predicted token for the next step in the output sequence.

### 3. Attention Mechanism 
- **Type**: Dot-product attention.
- **Function**: At each decoding step, attention computes a weighted sum of the encoder's hidden states based on their relevance to the current decoder hidden state.

## Experiments

To compare the performance of the two models (with and without attention), you can run experiments on your dataset. During training, you will see the training and validation loss, as well as the model's performance metrics (e.g., BLEU score for translation tasks).

### Example Results:

| Model               | Training Time | Validation Loss | BLEU Score |
|---------------------|---------------|-----------------|------------|
| Seq2Seq (No Attention) | 3 hours       | 1.24            | 30.1       |
| Seq2Seq (Attention)    | 4.5 hours     | 0.98            | 38.6       |
