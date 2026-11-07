import hashlib
import os

SECRET = os.environ.get("SECRET", "dev")


def hash_password(password: str) -> str:
    return hashlib.sha256((password + SECRET).encode()).hexdigest()


class AuthService:
    def __init__(self, users: dict[str, str]) -> None:
        self.users = users

    def login(self, name: str, password: str) -> bool:
        stored = self.users.get(name)
        return stored == hash_password(password)

    @staticmethod
    def is_strong(password: str) -> bool:
        return len(password) >= 12

    class Policy:
        def max_attempts(self) -> int:
            return 5


def audit_report(events: list[str]) -> list[str]:
    lines = []
    lines.append("AUDIT REPORT")
    lines.append("============")
    for event in events:
        if event.startswith("login"):
            lines.append(f"LOGIN: {event}")
        elif event.startswith("logout"):
            lines.append(f"LOGOUT: {event}")
        else:
            lines.append(f"OTHER: {event}")
    lines.append("============")
    lines.append(f"TOTAL: {len(events)}")
    return lines
