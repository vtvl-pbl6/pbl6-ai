from flask import Blueprint, request

from hate_speech_text_span_detection.model.model_utils import detection
from middleware.jwt_middleware import token_required
from utils.abstract_response import AppResponse, Errors


hate_speech_text_span_bp = Blueprint(
    "hate_speech_text_span", __name__, url_prefix="/api/v1/hate-speech-text-span"
)


@hate_speech_text_span_bp.route("", methods=["POST"])
@token_required
def seed(_):
    body = request.get_json()
    text = body.get("text")

    if not text:
        return AppResponse.error(Errors.TEXT_IS_REQUIRED, 400)

    text_span_detection_result = detection(text)
    return AppResponse.success_with_data(text_span_detection_result.to_dict())
