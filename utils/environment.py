import os

from dotenv import find_dotenv, load_dotenv


class Env:
    load_dotenv(find_dotenv())

    FLASK_ENV = os.environ.get("FLASK_ENV", "development").lower()
    API_HEADER_NAME = os.environ.get("API_HEADER_NAME", "example")
    API_HEADER_VALUE = os.environ.get("API_HEADER_VALUE", "example")
    DATABASE_URI = os.environ.get("DATABASE_URI", "")
