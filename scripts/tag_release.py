#!/usr/bin/env python3
"""Create a validated release tag from the VERSION file.

The VERSION file at the repo root is the single source of truth for the
project version. This script creates the matching annotated git tag
(v<VERSION>) after verifying that

  1. VERSION contains a valid non-dev release version (X.Y.Z),
  2. the working tree is clean (no uncommitted changes),
  3. docs/changelog.rst has an entry for this version,
  4. the tag does not already exist.

The tag <-> VERSION consistency is additionally enforced in CI by
.github/workflows/check_version.yml on every tag push.

Usage:
    pixi run tag-release          # or: python scripts/tag_release.py
    git push origin v<VERSION>    # afterwards, to publish the tag
"""

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASE_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
ANY_VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(\.dev[0-9]+)?$")


def fail(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main() -> None:
    # 1. read and validate VERSION
    version_file = REPO_ROOT / "VERSION"
    if not version_file.exists():
        fail("VERSION file not found at repo root")
    version = version_file.read_text(encoding="utf-8").strip()

    if not ANY_VERSION_RE.match(version):
        fail(f"'{version}' in the VERSION file is not a valid version "
             "(expected X.Y.Z or X.Y.Z.devN)")
    if not RELEASE_RE.match(version):
        fail(f"'{version}' is a dev version - bump the VERSION file to a "
             "release version (X.Y.Z) before tagging")

    tag = f"v{version}"

    # 2. working tree must be clean
    if git("status", "--porcelain"):
        fail("working tree is not clean - commit or stash your changes first")

    # 3. changelog must mention this version
    changelog = REPO_ROOT / "docs" / "changelog.rst"
    if not changelog.exists():
        fail(f"{changelog} not found")
    if tag not in changelog.read_text(encoding="utf-8"):
        fail(f"docs/changelog.rst has no entry for {tag} - add one first")

    # 4. tag must not exist yet
    existing = git("tag", "--list", tag)
    if existing:
        fail(f"tag {tag} already exists")

    # create the annotated tag
    git("tag", "-a", tag, "-m", f"release {tag}")
    print(f"created annotated tag {tag} on commit {git('rev-parse', '--short', 'HEAD')}")
    print(f"publish it with:  git push origin {tag}")
    print(f"afterwards, bump VERSION to the next dev version "
          f"(e.g. {bump_patch(version)}.dev0)")


def bump_patch(version: str) -> str:
    major, minor, patch = version.split(".")
    return f"{major}.{minor}.{int(patch) + 1}"


if __name__ == "__main__":
    main()
