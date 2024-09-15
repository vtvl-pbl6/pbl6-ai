from datetime import datetime
from sqlalchemy import TIMESTAMP, Column, BigInteger as Long, ForeignKey
from utils import get_instance

_, db = get_instance()


class Follower(db.Model):
    __tablename__ = "followers"

    id = Column("id", Long, nullable=False, primary_key=True, autoincrement=True)
    user_id = Column(
        "user_id",
        Long,
        ForeignKey("accounts.id", match="FULL"),
        nullable=False,
    )
    follower_id = Column(
        "follower_id",
        Long,
        ForeignKey("accounts.id", match="FULL"),
        nullable=False,
    )
    created_at = Column("created_at", TIMESTAMP, nullable=False)
    updated_at = Column("updated_at", TIMESTAMP)
    deleted_at = Column("deleted_at", TIMESTAMP)

    def __repr__(self):
        return f"<Follower {self.id}>"

    def __init__(
        self,
        user_id: int,
        follower_id: int,
    ):
        self.user_id = user_id
        self.follower_id = follower_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.deleted_at = None
