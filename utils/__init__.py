import logging
import sys

from flask import Flask
from flask_cors import CORS

from utils.environment import Env

app = None


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)

    return app


def get_instance() -> Flask:
    global app
    if app:
        return app

    app = create_app()
    return app


def setup_logger() -> logging.Logger:
    logger = logging.getLogger()
    for handler in logger.handlers:
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(f"%(levelname)s - %(funcName)s || %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger
