from flask import jsonify
from utils import setup_logger
from utils.constants import Errors, get_error_message


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
