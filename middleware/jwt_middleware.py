from functools import wraps
import jwt
from jwt.exceptions import ExpiredSignatureError, DecodeError, InvalidTokenError
from flask import request
from entities.account import Account
from utils import get_instance
from utils.abstract_response import AppResponse, Errors

_, db = get_instance()
public_key_path = "secrets/access_token_public_key.pem"


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if "Authorization" in request.headers:
            token = request.headers["Authorization"].split(" ")[1]
        if not token:
            return AppResponse.error(Errors.UNAUTHORIZED, 401)

        try:
            # Avoid name conflict by renaming the file object variable
            with open(public_key_path, "rb") as pub_key_file:
                public_key = pub_key_file.read()

            # Decode JWT using the public key
            jwt_claims = jwt.decode(token, public_key, algorithms=["RS256"])
            account_id = jwt_claims["sub"]

            # Query account from the database
            account = Account.query.filter_by(id=account_id).first()
            if account is None:
                return AppResponse.error(Errors.UNAUTHORIZED, 401)

        except ExpiredSignatureError:
            return AppResponse.error(Errors.EXPIRED_TOKEN, 401)

        except InvalidTokenError or DecodeError:
            return AppResponse.error(Errors.INVALID_TOKEN, 401)

        except Exception as e:
            return AppResponse.server_error(e)

        # Pass the account object to the decorated function
        return f(account, *args, **kwargs)

    return decorated
