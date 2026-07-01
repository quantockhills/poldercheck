from pathlib import Path


def pytest_configure(config):
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        import os

        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
