from utils.abstract_response import AppResponse
from flask import Blueprint

user_bp = Blueprint("user", __name__, url_prefix="/api/v1/user")


@user_bp.route("/hello", methods=["GET"])
def hello_world():
    return AppResponse.success_with_data(data="Hello World")
