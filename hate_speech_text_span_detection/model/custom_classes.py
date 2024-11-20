import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import List


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiTaskModel(nn.Module):
    def __init__(self, input_model):
        super(MultiTaskModel, self).__init__()
        self.bert = input_model
        self.dropout = nn.Dropout(0.1)
        self.positional_encoding = PositionalEncoding(
            d_model=self.bert.config.hidden_size
        )
        self.span_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        output = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )
        last_hidden_state = output[0]  # Shape: (batch_size, seq_length, hidden_size)

        # Add positional encoding
        last_hidden_state = self.positional_encoding(last_hidden_state)

        # Apply dropout
        last_hidden_state = self.dropout(last_hidden_state)

        # Apply span classifier
        span_logits = self.span_classifier(last_hidden_state)

        span_logits = span_logits.permute(0, 2, 1)
        span_logits = torch.sigmoid(span_logits)
        span_logits = span_logits.permute(0, 2, 1)

        return span_logits

    def predict(self, input_ids, attention_mask, threshold: float) -> np.ndarray[tuple]:
        span_logits = self.forward(input_ids, attention_mask)  # Tensor
        span_logits = (
            span_logits.squeeze().cpu().detach().numpy()
        )  # Convert tensor to numpy array

        span_logits = (
            span_logits.reshape(1, 64) if len(span_logits.shape) == 1 else span_logits
        )

        span_logits = [max(e) for e in span_logits]

        # Convert logits to binary labels (batch_size, 1)
        predictions = [(1, e) if e > threshold else (0, e) for e in span_logits]

        return predictions


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
    def __init__(self, word: str, start: int, end: int, score: float = 0.0):
        self.word = word
        self.start = start
        self.end = end
        self.score = score

    def __str__(self):
        return f"Word: {self.word}, Start: {self.start}, End: {self.end}, Score: {self.score}"

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "score": self.score,
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
