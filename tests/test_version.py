"""Consistency between the VERSION file (single source of truth) and the
version reported by the compiled extension module."""

import re
from pathlib import Path

import pytest
import parallelproj_core as pp

_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


@pytest.mark.skipif(
    not _VERSION_FILE.exists(),
    reason="VERSION file not present (not running from a repo checkout)",
)
def test_version_matches_version_file():
    version = _VERSION_FILE.read_text(encoding="utf-8").strip()

    # VERSION file must be a valid version
    match = re.match(r"^([0-9]+)\.([0-9]+)\.([0-9]+)(\.dev[0-9]+)?$", version)
    assert match, f"invalid version '{version}' in VERSION file"

    # full version string must match exactly
    assert pp.__version__ == version

    # numeric components must match as well
    assert pp.version_major == int(match.group(1))
    assert pp.version_minor == int(match.group(2))
    assert pp.version_patch == int(match.group(3))
