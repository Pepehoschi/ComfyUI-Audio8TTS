# -*- coding: utf-8 -*-
"""Publish a filtered copy of this repository to the public GitHub repo.

The full working repository (code plus internal planning docs, tests,
benchmarks and API example graphs) lives in the private remote
"internal". The public repository must only ever receive the filtered
tree produced here - never push a branch to the public URL directly.

What it does:
  1. Clone the local repository into a throw-away directory.
  2. Strip all internal paths from the full history with git filter-repo.
  3. Force-push the filtered master to the public repository.

Run from anywhere:  python scripts/publish_public.py
Requires: git and "uvx git-filter-repo" on PATH.
"""

import pathlib
import shutil
import subprocess
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
PUBLIC_URL = "https://github.com/Pepehoschi/ComfyUI-Audio8TTS"
PUBLIC_BRANCH = "master"

# Keep this list in sync with what must never appear in the public repo.
EXCLUDE = [
    "PROGRESS.md",
    "ImplementationPlan.md",
    "TODO.MD",
    "IDEA.md",
    "NATIVE_OPS_PLAN.md",
    "tests",
    "benchmarks",
    "api_examples",
]

FILTER_REPO = ["uvx", "git-filter-repo"]


def run(cmd, cwd):
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(cwd), check=True)


def main():
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="audio8_publish_"))
    work = tmp / "repo"
    try:
        run(["git", "clone", "--no-hardlinks", REPO, work], cwd=REPO)

        args = FILTER_REPO + ["--invert-paths"]
        for path in EXCLUDE:
            args += ["--path", path]
        run(args + ["--force"], cwd=work)

        # Fail closed: refuse to publish if an excluded path survived.
        listing = subprocess.run(
            ["git", "ls-files"],
            cwd=str(work),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        leaked = [
            line
            for line in listing.splitlines()
            if line in EXCLUDE or any(line.startswith(p + "/") for p in EXCLUDE)
        ]
        if leaked:
            print("REFUSING TO PUBLISH, leaked paths:", leaked, file=sys.stderr)
            return 1

        run(
            ["git", "push", "--force", PUBLIC_URL, f"{PUBLIC_BRANCH}:{PUBLIC_BRANCH}"],
            cwd=work,
        )
        print(f"Published filtered {PUBLIC_BRANCH} to {PUBLIC_URL}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
