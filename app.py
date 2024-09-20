from utils import get_instance
from utils.environment import Env
from utils.abstract_response import AppResponse, Errors
from blueprints.seeder import seeder_bp
from blueprints.hate_speech_text_span import hate_speech_text_span_bp
from flask import request

app, _ = get_instance()

# Register blueprints
app.register_blueprint(seeder_bp)
app.register_blueprint(hate_speech_text_span_bp)


@app.before_request
def before_request_callback():
    if Env.API_HEADER_NAME in request.headers:
        if request.headers.get(Env.API_HEADER_NAME) != Env.API_HEADER_VALUE:
            return AppResponse.error(Errors.API_KEY_INVALID, 403)
    else:
        return AppResponse.error(Errors.API_KEY_NOT_FOUND, 401)


# Handle exceptions
@app.errorhandler(404)
def page_not_found(e):
    return AppResponse.error(Errors.PAGE_NOT_FOUND, 404)


@app.errorhandler(Exception)
def handle_error(e):
    return AppResponse.server_error(e)


if __name__ == "__main__":
    app.run(
        debug=False if Env.FLASK_ENV == "production" else True, port=8081, threaded=True
    )
