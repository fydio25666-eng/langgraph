"""Container health probe for the FastAPI service."""
from urllib.error import URLError
from urllib.request import urlopen


def main() -> None:
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=3) as response:
            if response.status != 200:
                raise SystemExit(1)
    except (OSError, URLError):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
