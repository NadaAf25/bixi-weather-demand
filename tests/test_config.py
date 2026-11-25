"""
test_config.py
Checks that the configuration is really central and that the environment is pinned.
No internet, no database, no large file needed.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = sorted(SRC.glob("[0-9][0-9]_*.py"))

# Settings that must live in src/config.py only
FORBIDDEN = {
    r"^(SEED|TARGET|CALENDAR|WEATHER|ALPHA)\s*=": "constant defined outside config.py",
    r"max_iter\s*=": "model settings must come from GB_PARAMS",
    r"learning_rate\s*=": "model settings must come from GB_PARAMS",
    r"Path\(__file__\)": "paths must come from config.py",
    r"\"data\"\s*/\s*\"processed\"": "paths must come from config.py",
    r"def conformal_margin": "shared helper lives in conformal.py",
}


def test_all_sixteen_scripts_found():
    assert [p.name[:2] for p in SCRIPTS] == [f"{i:02d}" for i in range(1, 17)]

def test_every_script_imports_the_config():
    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^from config import", text, re.M), f"{path.name} ignores config.py"


def test_no_setting_is_copied_into_a_script():
    for path in SCRIPTS:
        text = path.read_text(encoding="utf-8")
        for pattern, reason in FORBIDDEN.items():
            assert not re.search(pattern, text, re.M), f"{path.name}: {reason} ({pattern})"


def test_dependencies_are_pinned():
    for name in ("requirements.txt", "requirements-dev.txt"):
        lines = [l.strip() for l in (ROOT / name).read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.strip().startswith("#")]
        assert lines, f"{name} is empty"
        for line in lines:
            assert line.startswith("-r ") or re.fullmatch(r"[A-Za-z0-9_.\-]+==[0-9][^\s]*", line), \
                f"{name}: '{line}' is not pinned with =="


def test_config_paths_point_to_the_repository_files():
    sys.path.insert(0, str(SRC))
    try:
        import config
    finally:
        sys.path.remove(str(SRC))
    assert config.ROOT == ROOT
    assert config.SCHEMA_SQL.exists()
    assert config.FEATURES_CSV.exists()
    assert config.SCENARIO_INTERVALS_CSV.exists()
    assert len(config.TEMPS) * len(config.RAIN_LEVELS) * 24 * 7 == 16_632