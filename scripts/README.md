# Scripts

Benchmarks, profiling drivers and maintenance scripts. They are not part of the library or its
tests. Run them from the repository root, with the build of `parallelproj_core` on `PYTHONPATH`,
e.g. on a GPU system with the CUDA 13 build:

```bash
pixi run -e cuda13 build
PYTHONPATH=build-cuda13 pixi run -e cuda13 python scripts/<script>.py --help
```

All benchmarks have a `--device` option (`cuda:0`, `cpu`) and a `--preset` option for the problem
size. The `small` preset is a quick check on the CPU. Every script describes what it measures, its
options and example calls in its docstring (`--help`).

## Benchmarks

| Script | What it measures | Used in |
|---|---|---|
| `benchmark_tof_sinogram.py` | TOF sinogram forward / back projection of one image vs. the non-TOF projection along the same LORs. Presets for SIGNA-like geometries (incl. one OSEM subset of the full sinogram, `signa_subset`), CTR, TOF bin width; LOR order and image layout (`--lor_order`, `--axial_axis`) | [Performance](https://libparallelproj.readthedocs.io/en/latest/performance.html), [TOF sinogram vs. TOF listmode](https://libparallelproj.readthedocs.io/en/latest/theory/sino_vs_lm.html) |
| `benchmark_tof_listmode.py` | TOF listmode forward / back projection for different event orders (random, by LOR, Morton codes of the TOF centers, ...), threads per block, Gaussian vs. tabulated kernel, and the cost of sorting inside every call | #54, [Performance](https://libparallelproj.readthedocs.io/en/latest/performance.html) |
| `benchmark_tof_listmode_scaling.py` | TOF listmode projections vs. the number of events (e.g. 1e6 ... 3e8), random vs. sorted (`tof_lm_sort_key`), incl. the time for sorting; large event numbers in chunks | [TOF sinogram vs. TOF listmode](https://libparallelproj.readthedocs.io/en/latest/theory/sino_vs_lm.html) |
| `benchmark_batch.py` | B data sets sharing one projector (a mini batch in a network): TOF / non-TOF projections and a Poisson log-likelihood gradient + Hessian-vector "layer", one call per data set (and batched calls, if a batched API exists) | #47 (batched projectors, deferred) |
| `benchmark_v2_vs_v3.py` | Speed **and results** of two library versions (e.g. 2.0.7 vs. 3.0): `run` once per version (with that version's build on `PYTHONPATH`), then `compare` the two result files (speed-ups and max. relative differences) | release checks |

## Profiling

| Script | Purpose |
|---|---|
| `profile_nontof_sinogram.py` | Minimal driver for Nsight Compute: one warm-up and one profiled call of `joseph3d_fwd` and `joseph3d_back` for a preset of `benchmark_tof_sinogram.py` (#53). Prints wall-clock times, so it also runs without the profiler. |

## Maintenance

| Script | Purpose |
|---|---|
| `run_examples.py` | Runs every gallery example as a standalone script (pass / fail + run time), incl. the GPU-only examples. Pixi tasks: `pixi run -e docs run-examples` (CPU build), `pixi run run-examples-gpu` (CUDA build). |
| `tag_release.py` | Creates the release tag `v<VERSION>` after checking VERSION, a clean tree, the changelog entry and that the tag does not exist yet (`pixi run tag-release`). |

## Interpreting GPU timings

- GPU calls can be asynchronous: the benchmarks synchronize the device before reading the clock,
  and report the best of several calls after a warm-up call.
- Results depend on the GPU, the driver and the problem size; the numbers in the documentation were
  measured on a desktop GPU (NVIDIA RTX A4500) and are meant as a guide, not as a guarantee.
