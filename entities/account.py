from datetime import datetime
from sqlalchemy import TIMESTAMP, Column, BigInteger as Long, Enum, String
from utils import get_instance
from entities.enumerated_model.account_role import AccountRole


_, db = get_instance()


class Account(db.Model):
    __tablename__ = "accounts"

    id = Column("id", Long, nullable=False, primary_key=True, autoincrement=True)
    email = Column("email", String(255), nullable=False, unique=True)
    password = Column("password", String(255), nullable=False)
    fullName = Column("full_name", String(255))
    phone = Column("phone", String(255))
    address = Column("address", String(255))
    avatarUrl = Column("avatar_url", String(255))
    role = Column("role", Enum(AccountRole), nullable=False)
    created_at = Column("created_at", TIMESTAMP, nullable=False)

    def __repr__(self):
        return f"<Account {self.id}>"

    def __init__(
        self,
        email: str,
        password: str,
        fullName: str,
        phone: str,
        address: str,
        avatarUrl: str,
        role: AccountRole,
    ):
        self.email = email
        self.password = password
        self.fullName = fullName
        self.phone = phone
        self.address = address
        self.avatarUrl = avatarUrl
        self.role = role
        self.created_at = datetime.now()
