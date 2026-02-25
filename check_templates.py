import sys
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, TemplateSyntaxError

def main():
    templates_dir = Path(__file__).parent / 'Discord Bot' / 'db_viewer' / 'templates'
    if not templates_dir.exists():
        print(f"Error: {templates_dir} does not exist.")
        sys.exit(1)

    env = Environment(loader=FileSystemLoader(str(templates_dir)))
    env.filters['kyiv_time'] = lambda x, *args, **kwargs: x
    env.filters['json_pretty'] = lambda x, *args, **kwargs: x
    errors = 0

    for template_path in templates_dir.rglob('*.html'):
        rel_path = template_path.relative_to(templates_dir).as_posix()
        try:
            env.get_template(rel_path)
            # print(f"OK: {rel_path}")
        except TemplateSyntaxError as e:
            print(f"Syntax Error in {rel_path}:{e.lineno} - {e.message}")
            errors += 1
        except Exception as e:
            print(f"Other Error parsing {rel_path}: {e}")
            errors += 1

    if errors > 0:
        print(f"Found {errors} template errors.")
        sys.exit(1)
    else:
        print("All templates parsed successfully.")
        sys.exit(0)

if __name__ == '__main__':
    main()
