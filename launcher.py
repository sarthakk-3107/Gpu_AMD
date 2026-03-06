"""
Launcher script for running benchmarks across multiple processes.
Supports single-node multi-GPU and multi-node setups.

Usage:
    # Single node, 4 GPUs, NCCL backend
    python launcher.py --nproc 4 --backend nccl

    # Single node, 2 processes, Gloo backend (CPU-only)
    python launcher.py --nproc 2 --backend gloo

    # Compare both backends
    python launcher.py --nproc 2 --compare-backends
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def launch_benchmark(nproc, backend, output_dir="results"):
    """Launch distributed benchmark using torchrun."""
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", str(nproc),
        "--master_addr", "127.0.0.1",
        "--master_port", "29500",
        "benchmark.py",
        "--backend", backend,
        "--output-dir", output_dir,
    ]

    print(f"\n{'='*60}")
    print(f"Launching benchmark: {backend} backend, {nproc} processes")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"Benchmark failed with return code {result.returncode}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Launch distributed network benchmarks")
    parser.add_argument("--nproc", type=int, default=2,
                        help="Number of processes per node (default: 2)")
    parser.add_argument("--backend", choices=["gloo", "nccl"], default="gloo",
                        help="Distributed backend (default: gloo)")
    parser.add_argument("--compare-backends", action="store_true",
                        help="Run benchmarks with both gloo and nccl backends")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory (default: results)")
    args = parser.parse_args()

    if args.compare_backends:
        backends = ["gloo"]
        # Only add NCCL if CUDA is available
        try:
            import torch
            if torch.cuda.is_available():
                backends.append("nccl")
            else:
                print("CUDA not available, skipping NCCL backend")
        except ImportError:
            print("PyTorch not installed, skipping NCCL check")

        for backend in backends:
            success = launch_benchmark(args.nproc, backend, args.output_dir)
            if not success:
                print(f"Warning: {backend} benchmark failed, continuing...")
    else:
        launch_benchmark(args.nproc, args.backend, args.output_dir)

    print(f"\nDone! View results with: streamlit run dashboard.py")


if __name__ == "__main__":
    main()
