import os

from dotenv import find_dotenv, load_dotenv


class Env:
    load_dotenv(find_dotenv())

    FLASK_ENV = os.environ.get("FLASK_ENV", "development").lower()
    API_HEADER_NAME = os.environ.get("API_HEADER_NAME", "")
    API_HEADER_VALUE = os.environ.get("API_HEADER_VALUE", "")

    # Database
    DATABASE_HOST = (
        f"{os.environ.get("DATABASE_HOST", "")}:{os.environ.get("DATABASE_PORT", "")}"
        if os.environ.get("DATABASE_PORT", "")
        else os.environ.get("DATABASE_HOST", "")
    )
    DATABASE_NAME = os.environ.get("DATABASE_NAME", "")
    DATABASE_USERNAME = os.environ.get("DATABASE_USERNAME", "")
    DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD", "")
    DATABASE_URI = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}/{DATABASE_NAME}"  # postgresql://<username>:<password>@localhost/<database_name>
