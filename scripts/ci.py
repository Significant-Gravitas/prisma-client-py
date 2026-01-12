import os
import sys
import subprocess

PYTHON_MAPPING = {
    '3.10': '3.10',
    '3.11': '3.11',
    '3.12': '3.12',
    '3.13': '3.13',
}


def main() -> None:
    target = os.environ['TARGET_PYTHON']
    python_version = PYTHON_MAPPING.get(target, target)
    subprocess.check_call(['nox', '-p', python_version, *sys.argv[1:]])


if __name__ == '__main__':
    main()
