from typing import List
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
from io import BytesIO
from nsfw_detection.custom_classes import DetectionResult, NSFWDetectionResult
from utils import get_path
from utils.environment import Env

model = None
img_height = 180
img_width = 180
class_names = ["safe", "unsafe"]


def get_model(checkpoint_path: str = "epoch_final.pt"):
    global model
    if model is not None:
        return model

    model = tf.keras.models.load_model(checkpoint_path)

    return model


def load_image(url: str):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((180, 180))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def detection(img_urls: List[str], threshold: float = 0.7) -> NSFWDetectionResult:
    if not img_urls:
        return []

    model = get_model(
        checkpoint_path=get_path(
            "nsfw_detection/checkpoint",
            Env.NSFW_IMAGE_MODEL_NAME,
        )
    )

    results = []
    for img_url in img_urls:
        try:
            img = load_image(img_url)
            predictions = model.predict(img)
            probabilities = tf.keras.activations.softmax(
                predictions
            ).numpy()  # Apply softmax to get probabilities

            # Convert probabilities to float for JSON serialization
            probability = float(probabilities[:, -1][0])

            if probability > threshold:
                results.append(DetectionResult(img_url, probability))
        except Exception as e:
            results.append(DetectionResult(img_url, message=e))

    return NSFWDetectionResult(results)
