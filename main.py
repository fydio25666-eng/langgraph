"""Command-line entry point for the LangGraph customer-support demo."""
import sys

from graph import app
from nodes import human_fallback


def main() -> None:
    try:  # Keep unexpected graph failures as a structured user-facing response.
        output = app.invoke(
            {"question": " ".join(sys.argv[1:])},
            config={"configurable": {"thread_id": "cli-default"}},
        )
        print(output["result"].model_dump_json(indent=2))
    except Exception:
        print(human_fallback("系统异常").model_dump_json(indent=2))


if __name__ == "__main__":
    main()
