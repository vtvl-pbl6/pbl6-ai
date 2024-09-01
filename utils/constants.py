from enum import Enum


class Errors(Enum):
    INTERNAL_SERVER_ERROR = "internal_server_error"
    BAD_REQUEST_ERROR = "bad_request_error"
    PAGE_NOT_FOUND = "page_not_found"
    FORBIDDEN = "forbidden"
    UNAUTHORIZED = "unauthorized"
    API_KEY_NOT_FOUND = "api_key_is_required"
    API_KEY_INVALID = "api_key_not_match"


def get_error_message(error: Errors):
    return error_messages.get(error, "Unknown error")


# define error messages
error_messages = {
    Errors.INTERNAL_SERVER_ERROR: {
        "code": "ERR_SER0101",
        "message": "Internal server error",
    },
    Errors.BAD_REQUEST_ERROR: {
        "code": "ERR_SER0102",
        "message": "Bad request error",
    },
    Errors.PAGE_NOT_FOUND: {
        "code": "ERR_SER0103",
        "message": "Page not found",
    },
    Errors.FORBIDDEN: {
        "code": "ERR_AUTH0101",
        "message": "Access Denied! You don’t have permission to access!",
    },
    Errors.UNAUTHORIZED: {
        "code": "ERR_API0101",
        "message": "You do not have permission to access this data!",
    },
    Errors.API_KEY_NOT_FOUND: {
        "code": "ERR_AUTH0101",
        "message": "API key is required!",
    },
    Errors.API_KEY_INVALID: {
        "code": "ERR_API0102",
        "message": "You don't have permission to access this api!",
    },
}
