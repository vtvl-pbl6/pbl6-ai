from utils import get_instance
from utils.environment import Env
from utils.abstract_response import AppResponse
from utils.constants import Errors
from blueprints.user import user_bp
from flask import request

app, _ = get_instance()

app.register_blueprint(user_bp)


@app.before_request
def before_request_callback():
    if Env.API_HEADER_NAME in request.headers:
        if request.headers.get(Env.API_HEADER_NAME) != Env.API_HEADER_VALUE:
            return AppResponse.error(Errors.API_KEY_INVALID, 403)
    else:
        return AppResponse.error(Errors.API_KEY_NOT_FOUND, 401)


@app.errorhandler(404)
def page_not_found(e):
    return AppResponse.error(Errors.PAGE_NOT_FOUND, 404)


if __name__ == "__main__":
    app.run(
        debug=False if Env.FLASK_ENV == "production" else True, port=8081, threaded=True
    )
