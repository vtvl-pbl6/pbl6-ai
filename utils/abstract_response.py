from flask import jsonify
from utils import setup_logger
from enum import Enum
import yaml


class Errors(Enum):
    INTERNAL_SERVER_ERROR = "internal_server_error"
    BAD_REQUEST_ERROR = "bad_request_error"
    PAGE_NOT_FOUND = "page_not_found"
    FORBIDDEN = "forbidden"
    UNAUTHORIZED = "unauthorized"
    API_KEY_NOT_FOUND = "api_key_is_required"
    API_KEY_INVALID = "api_key_not_match"
    INVALID_TOKEN = "invalid_token"
    EXPIRED_TOKEN = "expired_token"


def get_error_message(error: Errors):
    # load error.yml file
    error_yml_file_path = "utils/resources/errors.yml"
    data = {}
    with open(error_yml_file_path, "r") as file:
        data = yaml.safe_load(file)

    return data.get(error.value, {"code": "ERR_SER0101", "message": "Unknown error"})


class AppResponse:
    @staticmethod
    def success_with_data(data: dict | list = {}, status_code: int = 200):
        return (
            jsonify(
                {
                    "is_success": True,
                    "data": data,
                }
            ),
            status_code,
        )

    @staticmethod
    def success_with_meta(meta: dict, data: dict | list = []):
        return (
            jsonify(
                {
                    "is_success": True,
                    "data": data,
                    "metadata": meta,
                }
            ),
            200,
        )

    @staticmethod
    def error(error: Errors, status_code: int = 400):
        return (
            jsonify(
                {
                    "is_success": False,
                    "errors": [get_error_message(error)],
                }
            ),
            status_code,
        )

    @staticmethod
    def server_error(error, status_code: int = 500):
        logger = setup_logger()
        logger.error(error)

        return (
            jsonify(
                {
                    "is_success": False,
                    "errors": [
                        {
                            "code": "ERR_SER0101",
                            "message": str(error),
                        }
                    ],
                }
            ),
            status_code,
        )
