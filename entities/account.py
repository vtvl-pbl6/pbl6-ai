from datetime import datetime
from sqlalchemy import TIMESTAMP, Column, BigInteger as Long, String, TEXT
from sqlalchemy.dialects.postgresql import ENUM
from utils import get_instance
from entities.enumerated_model import (
    AccountRole,
    AccountStatus,
    AccountGender,
    Visibility,
)


_, db = get_instance()


class Account(db.Model):
    __tablename__ = "accounts"

    id = Column("id", Long, nullable=False, primary_key=True, autoincrement=True)
    email = Column("email", String(255), nullable=False, unique=True)
    password = Column("password", String(255), nullable=False)
    firstName = Column("first_name", String(255), nullable=False)
    lastName = Column("last_name", String(255), nullable=False)
    status = Column(
        "status",
        ENUM(AccountStatus, name="account_status", create_type=False),
        nullable=False,
    )
    role = Column(
        "role",
        ENUM(AccountRole, name="account_role", create_type=False),
        nullable=False,
    )
    display_name = Column("display_name", String(255), nullable=False, unique=True)
    birthday = Column("birthday", TIMESTAMP)
    gender = Column(
        "gender",
        ENUM(AccountGender, name="account_gender", create_type=False),
    )
    bio = Column("bio", TEXT)
    avatar = Column("avatar", Long)
    visibility = Column(
        "visibility",
        ENUM(Visibility, name="visibility", create_type=False),
        nullable=False,
    )
    language = Column("language", String(10))
    created_at = Column("created_at", TIMESTAMP, nullable=False)
    updated_at = Column("updated_at", TIMESTAMP)
    deleted_at = Column("deleted_at", TIMESTAMP)

    def __repr__(self):
        return f"<Account {self.id}>"

    def __init__(
        self,
        email: str,
        password: str,
        firstName: str,
        lastName: str,
        role: AccountRole,
        display_name: str,
        status: AccountStatus = AccountStatus.ACTIVE,
        birthday: datetime = None,
        gender: AccountGender = None,
        bio: str = None,
        visibility: Visibility = Visibility.PUBLIC,
        language: str = "vi",
    ):
        self.email = email
        self.password = password
        self.firstName = firstName
        self.lastName = lastName
        self.role = role
        self.display_name = display_name
        self.status = status
        self.birthday = birthday
        self.gender = gender
        self.bio = bio
        self.avatar = None
        self.visibility = visibility
        self.language = language
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.deleted_at = None
