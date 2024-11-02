from flask import Blueprint, request
from nsfw_detection.model_utils import detection
from utils.abstract_response import AppResponse, Errors
from utils.environment import Env

nsfw_bp = Blueprint("nsfw", __name__, url_prefix="/api/v1/nsfw")


@nsfw_bp.route("", methods=["POST"])
def moderate():
    body = request.get_json()
    img_urls = body.get("img_urls")
    key = body.get("key")

    if key != Env.MAIN_SERVER_KEY:
        return AppResponse.error(Errors.FORBIDDEN, 403)

    if not img_urls:
        return AppResponse.error(Errors.IMAGE_URL_IS_REQUIRED, 400)

    nsfw_detection_result = detection(img_urls)
    return AppResponse.success_with_data(nsfw_detection_result.to_dict())
