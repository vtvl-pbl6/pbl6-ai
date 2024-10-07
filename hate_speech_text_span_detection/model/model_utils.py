from typing import List
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import XLMRobertaModel, AutoTokenizer

from hate_speech_text_span_detection.preprocess.preprocess_utils import preprocess
from utils import get_path
from hate_speech_text_span_detection.model.custom_classes import (
    MultiTaskModel,
    PrepareData,
    SpanResult,
    TextSpanDetectionResult,
)

model = None
tokenizer = None
device = torch.device("cpu")


def get_tokenizer() -> AutoTokenizer:
    global tokenizer
    tokenizer = (
        AutoTokenizer.from_pretrained("xlm-roberta-base")
        if tokenizer is None
        else tokenizer
    )
    return tokenizer


def get_model(checkpoint_path: str = "epoch_final.pt") -> MultiTaskModel:
    global model
    if model is not None:
        return model

    # Define input model
    input_model = XLMRobertaModel.from_pretrained("xlm-roberta-base")
    tokenizer = get_tokenizer()
    input_model.resize_token_embeddings(len(tokenizer))

    # Load model
    model = MultiTaskModel(input_model)
    model.load_state_dict(
        torch.load(checkpoint_path, weights_only=False, map_location=device)
    )
    model.eval()

    return model


def create_dataloader(
    tokenizer: AutoTokenizer,
    batch_size: int,
    texts: List[str],
    max_len: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    dataset = PrepareData(tokenizer, texts, max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def detection(text: str, threshold: float = 0.5) -> TextSpanDetectionResult:
    # Get model and tokenizer
    model = get_model(
        checkpoint_path=get_path(
            "hate_speech_text_span_detection/model/checkpoint", "epoch_29.pt"
        )
    )
    tokenizer = get_tokenizer()

    # Preprocess text and tokenize
    words, indexes = preprocess(text)
    dataloader = create_dataloader(tokenizer, 64, words)
    prediction_results = np.array([])
    prediction_with_threshold_results = np.array([])
    for texts in dataloader:
        input_ids = texts["input_ids"].squeeze(1).to(device)
        attention_mask = texts["attention_mask"].to(device)
        predictions = model.predict(input_ids, attention_mask, threshold)
        # Add predictions to results
        prediction_results = np.append(
            prediction_results,
            [prediction for _, prediction in predictions],
        )
        prediction_with_threshold_results = np.append(
            prediction_with_threshold_results,
            [prediction_with_threshold for prediction_with_threshold, _ in predictions],
        )

    prediction_results = prediction_results.flatten()
    prediction_with_threshold_results = prediction_with_threshold_results.flatten()

    # Create span result
    span_result = []
    for i, (word, index) in enumerate(zip(words, indexes)):
        if prediction_with_threshold_results[i] == 1:
            span_result.append(
                SpanResult(
                    word=word,
                    start=index[0],
                    end=index[-1],
                    score=prediction_results[i],
                )
            )

    return TextSpanDetectionResult(text, span_result)
