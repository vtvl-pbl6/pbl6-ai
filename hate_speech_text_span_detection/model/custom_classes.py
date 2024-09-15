import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List, Tuple

from utils import tensor_to_numpy


# Model for text span detection
class MultiTaskModel(nn.Module):
    def __init__(self, input_model):
        super(MultiTaskModel, self).__init__()
        self.bert = input_model
        self.span_classifier = nn.Linear(768, 1)  # Classification head
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # Forward pass through the BERT model
        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        last_hidden_state = output[
            0
        ]  # Hidden state of shape (batch_size, sequence_length, hidden_size)

        # Apply dropout
        last_hidden_state = self.dropout(last_hidden_state)

        # Apply the span classifier to get logits for each token (batch_size, sequence_length, 1)
        span_logits = self.span_classifier(last_hidden_state)

        span_logits = span_logits.permute(0, 2, 1)
        span_logits = torch.sigmoid(span_logits)
        span_logits = span_logits.permute(0, 2, 1)

        return span_logits

    def predict(self, input_ids, attention_mask, threshold: float) -> np.ndarray:
        span_logits = self.forward(input_ids, attention_mask)

        # Convert logits to binary labels
        predictions = (
            span_logits > threshold
        ).float()  # (batch_size, sequence_length, 1)

        # Aggregate the predictions across the sequence
        aggregated_predictions = torch.mean(predictions, dim=1)  # (batch_size, 1)

        # If the average prediction is above the threshold, classify as HOS (1), otherwise NOT-HOS (0)
        final_predictions = (
            aggregated_predictions > threshold
        ).long()  # Final binary prediction (batch_size, 1)

        return tensor_to_numpy(final_predictions).flatten()


class PrepareData(Dataset):
    def __init__(
        self,
        tokenizer,
        texts: List[str],
        max_len: int = 64,
    ):
        # Tokenize text with output format {'input_ids': [], 'attention_mask': []}s
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_len,
                truncation=True,
                return_tensors="pt",
            )
            for text in texts
        ]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


class SpanResult:
    def __init__(self, word: str, start: int, end: int):
        self.word = word
        self.start = start
        self.end = end

    def __str__(self):
        return f"Word: {self.word}, Start: {self.start}, End: {self.end}"

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
        }


class TextSpanDetectionResult:
    def __init__(self, text: str, spans: List[SpanResult]):
        self.text = text
        self.spans = spans

    def __str__(self):
        return f"Text: {self.text}, HOS_Spans: {self.spans}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "hos_spans": [span.to_dict() for span in self.spans] if self.spans else [],
        }
