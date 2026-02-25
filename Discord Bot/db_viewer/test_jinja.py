from pathlib import Path

from jinja2 import Environment, FileSystemLoader


def main() -> int:
    base_dir = Path(__file__).resolve().parent
    templates_dir = base_dir / "templates"

    env = Environment(loader=FileSystemLoader(str(templates_dir), encoding="utf-8"))
    env.filters["kyiv_time"] = lambda x, *args, **kwargs: x
    env.filters["json_pretty"] = lambda x, *args, **kwargs: x

    has_error = False
    for path in sorted(templates_dir.glob("*.html")):
        try:
            env.get_template(path.name)
            print(f"OK: {path.name}")
        except Exception as exc:
            print(f"FAIL: {path.name} -> {exc}")
            has_error = True

    if has_error:
        return 1
    print("All templates parsed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
