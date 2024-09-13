from entities.account import Account
from middleware.jwt_middleware import token_required
from utils.abstract_response import AppResponse
from flask import Blueprint

user_bp = Blueprint("user", __name__, url_prefix="/api/v1/user")


@user_bp.route("/hello", methods=["POST"])
@token_required
def hello_world(account: Account):
    return AppResponse.success_with_data(data="Hello World")
