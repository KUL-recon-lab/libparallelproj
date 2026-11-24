"""Run every gallery example in docs/examples as a standalone script.

Each example runs in its own Python process (non-interactive matplotlib backend),
with the parallelproj_core module of the given CMake build directory and the
example helpers (docs/examples/utils.py, gpu_utils.py) on the Python path. Prints
pass / fail and the run time of every example, and the output of the failed ones.
Exits with 1 if an example fails.

Unlike the Sphinx gallery build, this also runs the examples with "gpu" in their
file name. Used by the pixi tasks ``run-examples`` (docs env, CPU build) and
``run-examples-gpu`` (docs-gpu env, CUDA build):

    pixi run -e docs run-examples
    pixi run run-examples-gpu

or directly, e.g. with the module of the cuda13 environment:

    pixi run -e cuda13 python scripts/run_examples.py --build-dir build-cuda13 -k getting_started
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "docs" / "examples"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build-dir", default="docs/_build_doxygen",
                        help="CMake build directory with the parallelproj_core module (relative to the repo root)")
    parser.add_argument("-k", "--filter", default="", help="only run examples whose path contains this string")
    parser.add_argument("--require-cuda", action="store_true",
                        help="fail unless parallelproj_core was built with CUDA")
    parser.add_argument("--timeout", type=float, default=1800.0, help="timeout per example in seconds")
    args = parser.parse_args()

    build_dir = (REPO / args.build_dir).resolve()
    env = dict(os.environ, MPLBACKEND="Agg",
               PYTHONPATH=os.pathsep.join([str(build_dir), str(EXAMPLES), os.environ.get("PYTHONPATH", "")]))

    # which parallelproj_core is used (and is it a CUDA build?)
    info = subprocess.run([sys.executable, "-c", "import parallelproj_core as p; print(p.__file__, p.cuda_enabled)"],
                          env=env, capture_output=True, text=True)
    if info.returncode != 0:
        print(f"cannot import parallelproj_core from {build_dir}:\n{info.stderr}")
        return 1
    module_file, cuda_enabled = info.stdout.split()
    print(f"parallelproj_core: {module_file} (cuda_enabled = {cuda_enabled})")
    if args.require_cuda and cuda_enabled != "1":
        print("--require-cuda: this parallelproj_core is not a CUDA build")
        return 1

    # the examples: numbered files (NN_*.py) in docs/examples and its subsections
    examples = sorted(p for p in EXAMPLES.rglob("[0-9][0-9]*_*.py")
                      if "auto_examples" not in p.parts and args.filter in str(p.relative_to(EXAMPLES)))
    if not examples:
        print("no examples found")
        return 1

    failed = []
    for path in examples:
        rel = path.relative_to(EXAMPLES)
        t0 = time.perf_counter()
        try:
            res = subprocess.run([sys.executable, path.name], cwd=path.parent, env=env, capture_output=True,
                                 text=True, timeout=args.timeout)
            ok, out = res.returncode == 0, res.stdout + res.stderr
        except subprocess.TimeoutExpired as e:
            ok, out = False, f"timeout after {args.timeout} s\n{e.stdout or ''}{e.stderr or ''}"
        dt = time.perf_counter() - t0
        print(f"{'ok  ' if ok else 'FAIL'} {dt:7.1f} s  {rel}", flush=True)
        if not ok:
            failed.append((rel, out))

    for rel, out in failed:
        print(f"\n{'=' * 80}\n{rel}\n{'=' * 80}\n{out[-5000:]}")
    print(f"\n{len(examples) - len(failed)} of {len(examples)} examples ran without errors")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
