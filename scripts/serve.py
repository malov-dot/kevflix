"""Process S: the KevFLIX app server, started by the app guard with `python -I -B scripts/serve.py --config <file>`.

-I keeps the interpreter isolated from user site/PYTHONPATH, so this script puts the install root on sys.path
itself, then runs relay.server.main() with the given arguments (the user's profile config).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from relay.server import main  # noqa: E402

sys.argv = [str(Path(__file__).resolve())] + sys.argv[1:]
main()
