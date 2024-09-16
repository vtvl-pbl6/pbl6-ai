import random
from typing import Tuple

from sqlalchemy import func, not_
from entities.account import Account
from entities.enumerated_model import AccountGender, AccountRole, AccountStatus
from utils import get_instance, setup_logger
from utils.abstract_response import AppResponse, Errors
from middleware.jwt_middleware import token_required
from flask import Blueprint, request
from entities.account import Account
from entities.follower import Follower
from entities.thread import Thread
from entities.thread_sharer import ThreadSharer
from faker import Faker

seeder_bp = Blueprint("seeder", __name__, url_prefix="/api/v1/seeder")

_, db = get_instance()


@seeder_bp.route("/", methods=["POST"])
@token_required
def seed(account: Account):
    if account.role != AccountRole.ADMIN:
        return AppResponse.error(Errors.FORBIDDEN, 403)

    body = request.get_json()
    reset = body.get("reset", False)
    repeat_times = body.get("repeat_times", 1000)

    try:
        if reset:
            Account.query.filter(
                not_(Account.email.in_(["user@gmail.com", "admin@gmail.com"]))
            ).delete(synchronize_session=False)
            Follower.query.delete()
            Thread.query.delete()
            ThreadSharer.query.delete()
            db.session.commit()

        seed_account(repeat_times)
        seed_follower(repeat_times)
        seed_thread(repeat_times)
    except Exception as e:
        return AppResponse.server_error(e)

    return AppResponse.success_with_data(data="Database seeded successfully!")


generated_display_name = set()  # To store already generated display names
generated_email = set()  # To store already generated emails


def get_unique_email_and_display_name() -> Tuple[str, str]:
    fake = Faker()
    email = None
    display_name = None

    while True:
        display_name = fake.profile(fields=["username"])["username"]
        if display_name not in generated_display_name:
            generated_display_name.add(display_name)
            break

    while True:
        email = fake.email()
        if email not in generated_email:
            generated_email.add(email)
            break

    return email, display_name


def seed_account(repeat_times: int):
    logger = setup_logger()
    fake = Faker()
    try:
        logger.info("Start seeding account...")

        for _ in range(repeat_times):
            ran_val = random.uniform(0, 1)
            email, display_name = get_unique_email_and_display_name()
            account = Account(
                email=email,
                password=fake.password(),
                firstName=fake.first_name(),
                lastName=fake.last_name(),
                role=AccountRole.USER,
                display_name=display_name,
                status=(
                    AccountStatus.ACTIVE if ran_val >= 0.3 else AccountStatus.INACTIVE
                ),
                birthday=fake.date_of_birth(),
                gender=(
                    AccountGender.MALE
                    if ran_val >= 0.7
                    else (
                        AccountGender.FEMALE if ran_val >= 0.3 else AccountGender.OTHER
                    )
                ),
                bio=fake.text(),
            )
            db.session.add(account)

        db.session.commit()
        logger.info("Finished seeding account...")
    except Exception as e:
        db.session.rollback()
        logger.error(e)
        raise e


def seed_follower(repeat_times: int):
    logger = setup_logger()
    try:
        logger.info("Start seeding follower...")

        for _ in range(repeat_times):
            account = Account.query.order_by(func.random()).first()
            follower = Account.query.order_by(func.random()).first()
            if account.id == follower.id:
                continue
            follower = Follower(user_id=account.id, follower_id=follower.id)
            db.session.add(follower)

        db.session.commit()
        logger.info("Finished seeding follower...")
    except Exception as e:
        db.session.rollback()
        logger.error(e)
        raise e


def seed_thread(repeat_times: int):
    logger = setup_logger()
    try:
        # Thread
        logger.info("Start seeding thread...")
        threads = []
        for _ in range(repeat_times):
            account = (
                Account.query.filter(Account.status != AccountStatus.INACTIVE)
                .order_by(func.random())
                .first()
            )
            thread = Thread(
                author_id=account.id,
                content=Faker().text(),
                reaction_num=random.randint(0, 1000),
                shared_num=0,
                is_pin=False,
            )
            db.session.add(thread)
            threads.append(thread)

        # ThreadSharer
        logger.info("Start seeding thread sharer...")
        thread_sharers = []
        for _ in range(repeat_times):
            account = (
                Account.query.filter(Account.status != AccountStatus.INACTIVE)
                .order_by(func.random())
                .first()
            )
            thread = threads[random.randint(0, len(threads) - 1)]
            thread_sharer = ThreadSharer(thread_id=thread.id, user_id=account.id)
            db.session.add(thread_sharer)
            thread_sharers.append(thread_sharer)

        # Update shared_num in Thread
        logger.info("Update shared_num in thread...")
        final_threads = []
        for thread in threads:
            thread.shared_num = len(
                list(
                    filter(
                        lambda thread_sharer: thread_sharer.thread_id == thread.id,
                        thread_sharers,
                    )
                )
            )
            db.session.add(thread)
            final_threads.append(thread)

        # Comment
        logger.info("Start seeding comment...")
        for _ in range(repeat_times):
            parent_thread = final_threads[random.randint(0, len(final_threads) - 1)]
            comment = final_threads[random.randint(0, len(final_threads) - 1)]

            if parent_thread.id == comment.id or comment.parent_thread_id is not None:
                continue

            comment.parent_thread_id = parent_thread.id
            db.session.add(comment)

        db.session.commit()
        logger.info("Finished seeding thread...")
    except Exception as e:
        db.session.rollback()
        logger.error(e)
        raise e
