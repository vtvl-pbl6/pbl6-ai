from datetime import datetime
from sqlalchemy import TIMESTAMP, Column, BigInteger as Long, ForeignKey
from utils import get_instance

_, db = get_instance()


class ThreadSharer(db.Model):
    __tablename__ = "thread_sharers"

    id = Column("id", Long, nullable=False, primary_key=True, autoincrement=True)
    thread_id = Column(
        "thread_id",
        Long,
        ForeignKey("threads.id", match="FULL"),
        nullable=False,
    )
    user_id = Column(
        "user_id",
        Long,
        ForeignKey("accounts.id", match="FULL"),
        nullable=False,
    )
    created_at = Column("created_at", TIMESTAMP, nullable=False)
    updated_at = Column("updated_at", TIMESTAMP)
    deleted_at = Column("deleted_at", TIMESTAMP)

    def __repr__(self):
        return f"<ThreadSharer {self.id}>"

    def __init__(
        self,
        thread_id: int,
        user_id: int,
    ):
        self.thread_id = thread_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.deleted_at = None
