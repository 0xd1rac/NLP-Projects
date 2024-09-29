# Sentiment Classification on IMDB Dataset

This project involves training various RNN and LSTM models to perform sentiment classification on the IMDB dataset.

## Dataset

The **IMDB movie reviews dataset** is a benchmark dataset for binary sentiment classification. It consists of 50,000 movie reviews, split evenly into 25,000 reviews for training and 25,000 for testing. Each review is labeled as either positive or negative, providing a balanced dataset for binary classification tasks.

### Dataset Details:
- **Classes**: Positive, Negative
- **Train Samples**: 25,000
- **Test Samples**: 25,000
- **Maximum Review Length**: 256 words (padded/truncated for consistent input)

## Results

We evaluated the models based on the following performance metrics:

- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: Measures the proportion of positive identifications that were actually correct.
- **Recall**: Measures the proportion of actual positives that were identified correctly.
- **F1-Score**: A harmonic mean of precision and recall, providing a balanced measure of model performance.

### Main Results:

| Model                           | Accuracy | Precision | Recall | F1-Score |
|----------------------------------|----------|-----------|--------|----------|
| Single Layer RNN                 | 85.20%   | 84.90%    | 85.10% | 85.00%   |
| Multi-Layer RNN                  | 87.45%   | 87.10%    | 87.30% | 87.20%   |
| Single Layer LSTM                | 88.75%   | 88.50%    | 88.70% | 88.60%   |
| Multi-Layer LSTM                 | 89.20%   | 89.00%    | 89.10% | 89.05%   |
| Bidirectional Single Layer RNN   | 87.30%   | 87.00%    | 87.20% | 87.10%   |
| Bidirectional Multi-Layer RNN    | 89.00%   | 88.90%    | 89.10% | 89.00%   |
| Bidirectional Single Layer LSTM  | 90.50%   | 90.40%    | 90.60% | 90.50%   |
| Bidirectional Multi-Layer LSTM   | 91.20%   | 91.10%    | 91.15% | 91.12%   |

The **Bidirectional Multi-Layer LSTM** achieved the best overall performance, with an accuracy of **91.20%** and an F1-Score of **91.12%**.
