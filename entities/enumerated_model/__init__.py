from enum import Enum


class AccountGender(Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    OTHER = "OTHER"


class AccountRole(Enum):
    USER = "USER"
    ADMIN = "ADMIN"


class AccountStatus(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class ThreadStatus(Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    PENDING = "PENDING"


class Visibility(Enum):
    PUBLIC = "PUBLIC"
    PRIVATE = "PRIVATE"
    FRIEND_ONLY = "FRIEND_ONLY"
