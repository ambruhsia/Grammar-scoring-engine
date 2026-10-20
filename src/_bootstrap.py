"""Re-run the calling script with the project's .venv interpreter when it isn't already in use.

Claude Code (the VS Code extension especially) may not start inside the activated venv, so
`python src/run.py` could otherwise hit a system Python without pandas or lightgbm.
Imported first by every entry script; does nothing when there is no .venv.
"""
import os
import subprocess
import sys
from pathlib import Path

_venv = Path(__file__).resolve().parents[1] / ".venv"
_py = _venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
_script = sys.argv[0] if sys.argv and sys.argv[0].endswith(".py") else ""
if (_script and _py.exists() and Path(sys.prefix).resolve() != _venv.resolve()
        and not os.environ.get("SHL_IN_VENV")):
    sys.exit(subprocess.call([str(_py), *sys.argv], env=dict(os.environ, SHL_IN_VENV="1")))
