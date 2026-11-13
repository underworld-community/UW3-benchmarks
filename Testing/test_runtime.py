"""
Run the benchmark scripts sequentially.

All benchmarks run with res=32 and 1 timestep where applicable.

Runs:
  1. Ex_AdvDiff_rotatingCone.py
  2. Ex_AdvDiff-hot_pipe.py
  3. Ex_Convection_BA_Cartesian_benchmark.py
  4. Ex_Convection_EBA_Cartesian_benchmark.py
  5. Ex_Convection_TALA_Cartesian_benchmark.py
  6. Ex_Darcy_1D_benchmark.py
  7. Ex_Diffusion_benchmark.py
  8. Ex_GW+temp_Coupling_benchmark.py
  9. Ex_ice_BM-sachau_expB_2D.py
 10. Ex_NL_Viscous_slabDetachment.py
 11. Ex_Stokes_Cartesian_SolCx.py
 12. Ex_Stokes_Sinker_benchmark.py
 13. Ex_stokes_sinkingBlock_benchmark.py
 14. Ex_VP_brick_benchmark.py
 15. Ex_VP_Spiegelman_Benchmark.py
 16. Ex_Convection_Annulus_benchmark_scaled.py

Usage:
    python test_runtime.py
    python test_runtime.py -np 4

Options:
    -np  <int>  Number of MPI processes (default: 1)
"""

import subprocess
import sys
import os

WORKING         = os.path.join(os.path.dirname(__file__), "..", "Working", "Cartesian")
WORKING_ANNULUS = os.path.join(os.path.dirname(__file__), "..", "Working", "Annulus")

_COMMON      = ["-uw_res", "32", "-uw_n_steps", "1"]
_HOTPIPE     = ["-uw_res", "32", "-uw_n_steps", "5"]
_RES_ONLY    = ["-uw_res", "32"]
_DARCY       = ["-uw_res", "20"]
_DIFFUSION   = ["-uw_res", "32", "-uw_n_steps", "1"]
_ICE         = ["-uw_res_x", "32", "-uw_res_y", "16"]
_SLAB        = ["-uw_res", "20", "-uw_n_steps", "1"]
_SOLCX       = ["-uw_res", "32"]
_SINKER      = ["-uw_res", "32", "-uw_n_steps", "1"]
_SINKBLOCK   = ["-uw_res", "32"]
_BRICK       = ["-uw_res_x", "64", "-uw_res_y", "16"]
_SPIEGELMAN  = ["-uw_problem_size", "1"]
_ANNULUS     = ["-uw_res", "0.3", "-uw_n_steps", "1"]
_CONV        = ["-uw_res", "16", "-uw_n_steps", "1"]

BENCHMARKS = [
    (os.path.join(WORKING, "Ex_AdvDiff_rotatingCone.py"),               _COMMON),
    (os.path.join(WORKING, "Ex_AdvDiff-hot_pipe.py"),                   _HOTPIPE),
    (os.path.join(WORKING, "Ex_Convection_BA_Cartesian_benchmark.py"),  _CONV),
    (os.path.join(WORKING, "Ex_Convection_EBA_Cartesian_benchmark.py"), _CONV),
    (os.path.join(WORKING, "Ex_Convection_TALA_Cartesian_benchmark.py"),_CONV),
    (os.path.join(WORKING, "Ex_Darcy_1D_benchmark.py"),                 _DARCY),
    (os.path.join(WORKING, "Ex_Diffusion_benchmark.py"),                _DIFFUSION),
    (os.path.join(WORKING, "Ex_GW+temp_Coupling_benchmark.py"),         _RES_ONLY),
    (os.path.join(WORKING, "Ex_ice_BM-sachau_expB_2D.py"),              _ICE),
    (os.path.join(WORKING, "Ex_NL_Viscous_slabDetachment.py"),          _SLAB),
    (os.path.join(WORKING, "Ex_Stokes_Cartesian_SolCx.py"),             _SOLCX),
    (os.path.join(WORKING, "Ex_Stokes_Sinker_benchmark.py"),            _SINKER),
    (os.path.join(WORKING, "Ex_stokes_sinkingBlock_benchmark.py"),      _SINKBLOCK),
    (os.path.join(WORKING, "Ex_VP_brick_benchmark.py"),                 _BRICK),
    (os.path.join(WORKING, "Ex_VP_Spiegelman_Benchmark.py"),            _SPIEGELMAN),
    (os.path.join(WORKING_ANNULUS, "Ex_Convection_Annulus_benchmark_scaled.py"), _ANNULUS),
]


def run(script, extra_args, np_procs):
    if np_procs > 1:
        cmd = ["mpirun", "-np", str(np_procs), "python", script] + extra_args
    else:
        cmd = ["python", script] + extra_args

    print(f"Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFAILED: {script} (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\nPASSED: {os.path.basename(script)}\n")


def main():
    np_procs = 1
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "-np":
            np_procs = int(args[i + 1])
            i += 2
        else:
            i += 1

    for script, extra_args in BENCHMARKS:
        run(script, extra_args, np_procs)

    print("All benchmarks passed.")


if __name__ == "__main__":
    main()
