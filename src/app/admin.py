import argparse
from getpass import getpass

from .storage.history import History, normalize_username


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grant admin access to an existing account or create one."
    )
    parser.add_argument("username")
    parser.add_argument(
        "--db",
        help="SQLite path; defaults to SECHA_CHAT_HISTORY_DB or .data/secha_chat.sqlite3",
    )
    args = parser.parse_args()
    history = History(args.db)
    history.initialize()
    with history._connect() as connection:
        row = connection.execute(
            "SELECT id FROM users WHERE normalized_username = ?",
            (normalize_username(args.username),),
        ).fetchone()
    if row is None:
        password = getpass("New admin password: ")
        if password != getpass("Confirm password: "):
            parser.error("Passwords do not match.")
        try:
            user_id = history.create_user(args.username, password).id
        except ValueError as exc:
            parser.error(str(exc))
    else:
        user_id = row["id"]
    with history._connect() as connection:
        connection.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user_id,))
    print(f"Admin access enabled for {args.username} in {history.db_path}.")


if __name__ == "__main__":
    main()
