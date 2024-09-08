import logging
import sys
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from utils.environment import Env
import os

app = None
db = None


def create_app() -> tuple[Flask, SQLAlchemy]:
    global app, db
    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = Env.DATABASE_URI
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    CORS(app)
    db = SQLAlchemy(app)

    return app, db


def get_instance() -> Flask:
    global db, app
    if app and db:
        return app, db

    app, db = create_app()
    return app, db


def get_path(
    parent_dir: str,
    file_name: str,
    mkdir_dir: bool = True,
    remove_file_if_exist: bool = False,
) -> str:
    parent_dir = f"{os.getcwd()}/{parent_dir}"
    file_path = f"{parent_dir}/{file_name}" if file_name else parent_dir

    if not os.path.exists(parent_dir) and mkdir_dir:
        os.makedirs(parent_dir)

    if os.path.exists(file_path) and remove_file_if_exist:
        os.remove(file_path)

    return file_path


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
