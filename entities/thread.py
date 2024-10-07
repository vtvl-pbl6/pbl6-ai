from datetime import datetime
from sqlalchemy import (
    TIMESTAMP,
    Column,
    BigInteger as Long,
    ForeignKey,
    TEXT,
    Integer,
    Boolean,
)
from entities.enumerated_model import ThreadStatus, Visibility
from sqlalchemy.dialects.postgresql import ENUM
from utils import get_instance

_, db = get_instance()


class Thread(db.Model):
    __tablename__ = "threads"

    id = Column("id", Long, nullable=False, primary_key=True, autoincrement=True)
    author_id = Column(
        "author_id",
        Long,
        ForeignKey("accounts.id", match="FULL"),
        nullable=False,
    )
    parent_thread_id = Column(
        "parent_thread_id",
        Long,
        ForeignKey("threads.id", match="FULL"),
    )
    content = Column("content", TEXT)
    reaction_num = Column("reaction_num", Integer, nullable=False)
    shared_num = Column("shared_num", Integer, nullable=False)
    is_pin = Column("is_pin", Boolean, nullable=False)
    status = Column(
        "status",
        ENUM(ThreadStatus, name="thread_status", create_type=False),
        nullable=False,
    )
    visibility = Column(
        "visibility",
        ENUM(Visibility, name="visibility", create_type=False),
        nullable=False,
    )
    created_at = Column("created_at", TIMESTAMP, nullable=False)
    updated_at = Column("updated_at", TIMESTAMP)
    deleted_at = Column("deleted_at", TIMESTAMP)

    def __repr__(self):
        return f"<Thread {self.id}>"

    def __init__(
        self,
        author_id: int,
        parent_thread_id: int = None,
        content: str = None,
        reaction_num: int = 0,
        shared_num: int = 0,
        is_pin: bool = False,
        status: ThreadStatus = ThreadStatus.CREATE_DONE,
        visibility: Visibility = Visibility.PUBLIC,
    ):
        self.author_id = author_id
        self.parent_thread_id = parent_thread_id
        self.content = content
        self.reaction_num = reaction_num
        self.shared_num = shared_num
        self.is_pin = is_pin
        self.status = status
        self.visibility = visibility
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.deleted_at = None
