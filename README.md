# GPU Cluster Network Bottleneck Analyzer

A Python-based profiling tool that benchmarks and visualizes network performance bottlenecks in distributed AI training workloads. It measures collective communication patterns (allreduce, allgather, broadcast), analyzes compute vs communication overlap, and generates data-driven infrastructure recommendations.

Built to understand how networking constraints impact large-scale ML training across GPU clusters.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)


## Why This Project?

Distributed AI training (think GPT-scale models across hundreds of GPUs) spends a significant chunk of time on inter-node communication rather than actual computation. Understanding where the bottleneck lies (compute-bound vs network-bound) is critical for making infrastructure decisions like:

- Should we upgrade to 400G InfiniBand?
- Is gradient compression worth the tradeoff?
- Are small allreduce operations killing our throughput?

This tool answers those questions with real data from your hardware.

---

## What It Measures

| Metric | Description |
|---|---|
| **Collective Bandwidth** | Throughput (Gbps) for allreduce, allgather, broadcast across 1MB to 256MB message sizes |
| **Latency Distribution** | Mean, P50, P99 latency per operation and message size |
| **Compute vs Communication** | Time split between matmul (forward pass) and gradient synchronization (allreduce) |
| **Bottleneck Classification** | Automatically flags whether workloads are compute-bound or network-bound |
| **Infrastructure Recommendations** | Product-style suggestions based on detected bottlenecks |

---

## Project Structure

```
gpu-network-analyzer/
├── benchmark.py        # Core benchmarking engine
├── dashboard.py        # Streamlit dashboard with Plotly visualizations
├── launcher.py         # Multi-process launcher via torchrun
├── requirements.txt    # Python dependencies
├── results/            # Auto-generated benchmark JSON outputs
└── docs/               # Screenshots for README
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/sarthak-adhangale/gpu-network-analyzer.git
cd gpu-network-analyzer
pip install -r requirements.txt
```

For lightweight setups (no GPU), install CPU-only PyTorch:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas numpy psutil plotly streamlit
```

### Run the Benchmark

**Standalone mode (works on any machine, no GPU required):**

```bash
python benchmark.py --output-dir results
```

**Simulate a larger cluster:**

```bash
python benchmark.py --output-dir results --world-size 8
```

**Multi-GPU mode (requires CUDA + multiple GPUs):**

```bash
python launcher.py --nproc 4 --backend nccl
```

**Compare Gloo vs NCCL backends:**

```bash
python launcher.py --nproc 2 --compare-backends
```

### Launch the Dashboard

```bash
python -m streamlit run dashboard.py
```

Opens at `http://localhost:8501`. The dashboard loads demo data automatically if no benchmark results exist.

---

## Dashboard Features

### System Overview
Displays GPU count, model, RAM, and CPU info detected from your machine.

### Bottleneck Analysis
Color-coded alerts (red/yellow/blue) identifying whether workloads are network-bound, compute-bound, or have small-message latency issues.

### Product Recommendations
Actionable infrastructure suggestions generated from benchmark data, including:
- Interconnect upgrade guidance (InfiniBand, RoCE)
- Gradient compression and async allreduce suggestions
- Gradient bucketing for small-message optimization
- NIC/MTU configuration checks

### Interactive Charts
- **Bandwidth vs Message Size**: Line charts per operation (allreduce, allgather, broadcast)
- **Latency vs Message Size**: Latency scaling across message sizes
- **P50 vs P99 Latency**: Tail latency distribution as grouped bar charts
- **Compute vs Communication Breakdown**: Stacked bar chart showing time split per matrix size
- **Backend Comparison**: Side-by-side Gloo vs NCCL performance (when both are benchmarked)

### Data Export
Download full benchmark report as JSON or collective benchmarks as CSV directly from the dashboard.

---

## How It Works

### Collective Operation Simulation

The tool simulates the three most common distributed training communication patterns:

- **Allreduce**: Each rank contributes a tensor, all ranks receive the sum. This is the primary operation used for gradient synchronization in data-parallel training.
- **Allgather**: Each rank sends its tensor to all other ranks. Common in model-parallel and pipeline-parallel setups.
- **Broadcast**: One rank sends its tensor to all others. Used for parameter initialization and checkpoint distribution.

For each operation, the benchmark sweeps across message sizes (1MB to 256MB) and measures latency, throughput, and variance.

### Compute vs Communication Overlap

Simulates a training iteration by measuring two phases:
1. **Compute phase**: Matrix multiplication representing a forward pass
2. **Communication phase**: Allreduce of gradient tensors across simulated ranks

The ratio between these two phases determines whether the workload is compute-bound or network-bound, which directly informs infrastructure decisions.

### Bottleneck Detection

The analysis engine automatically classifies bottlenecks based on three criteria:
- **Network-bound**: Communication time exceeds compute time for majority of configurations
- **Small message latency**: Average latency for messages <=4MB exceeds 1ms threshold, indicating overhead that can stall gradient sync in models with many small layers
- **Low bandwidth utilization**: Peak allreduce bandwidth below 10 Gbps, which is under typical datacenter-grade networking baselines

---

## Configuration

| Argument | Default | Description |
|---|---|---|
| `--output-dir` | `results` | Directory to save benchmark JSON |
| `--world-size` | `4` | Simulated number of GPUs/ranks |
| `--nproc` | `2` | Processes per node (launcher only) |
| `--backend` | `gloo` | Distributed backend: `gloo` or `nccl` (launcher only) |

---

## Sample Output

```
============================================================
GPU Cluster Network Bottleneck Analyzer
Mode: Standalone Simulation (world_size=4)
============================================================

System: DESKTOP-ABC123
  GPUs: 0, RAM: 16.0GB, CPUs: 8

Measuring network baseline...
  Ping: 0.0ms, BW: 8.52 Gbps

Benchmarking allreduce...
    allreduce @ 1MB: 3.245ms, 10.432 Gbps
    allreduce @ 64MB: 185.632ms, 11.045 Gbps
    allreduce @ 256MB: 742.118ms, 11.102 Gbps

Benchmarking compute-communication overlap...
    Overlap test @ 1024x1024 matrix...
    Overlap test @ 4096x4096 matrix...

============================================================
Results saved to results/benchmark_simulated_rank0.json
Run 'python -m streamlit run dashboard.py' to view the dashboard
============================================================
```

---

## Tech Stack

- **PyTorch**: Tensor operations and distributed communication primitives
- **Streamlit**: Interactive web dashboard
- **Plotly**: Data visualization (line charts, bar charts, tables)
- **Pandas/NumPy**: Data processing and statistical analysis
- **psutil**: System hardware profiling

---

## Future Improvements

- [ ] Add NCCL profiling with NVIDIA NCCL tests integration
- [ ] Support multi-node benchmarking over actual network interfaces
- [ ] Add RDMA/RoCE bandwidth measurement using perftest tools
- [ ] Integrate with NVIDIA Nsight Systems for GPU-level communication tracing
- [ ] Add ring-allreduce vs tree-allreduce topology comparison
- [ ] Export recommendations as a PDF report

---

## License

MIT

---

## Author

**Sarthak Adhangale**
- [Portfolio](https://sarthak-adhangale-portfolio.lovable.app/)
- [LinkedIn](https://linkedin.com/in/sarthak-adhangale)
- [GitHub](https://github.com/sarthak-adhangale)
