from functools import wraps
import jwt
from flask import request
from utils.abstract_response import AppResponse
from utils.constants import Errors

public_key_path = "/secrets/access_token_public_key.pem"


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]
        if not token:
            return AppResponse.error(Errors.UNAUTHORIZED, 401)
        try:
            with open(public_key_path, "rb") as f:
                public_key = f.read()
            data = jwt.decode(token, public_key, algorithms=["RS256"])
            user_id = data["id"]
            if user_id is None:
                return AppResponse.error(Errors.UNAUTHORIZED, 401)
        except Exception as e:
            return AppResponse.server_error(e)

        return f(user_id, *args, **kwargs)

    return decorated
