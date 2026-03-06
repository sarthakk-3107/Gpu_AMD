"""
GPU Cluster Network Bottleneck Analyzer - Streamlit Dashboard
Visualizes benchmark results with interactive charts and product-style recommendations.

Run: streamlit run dashboard.py
"""

import json
import glob
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GPU Cluster Network Analyzer",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
        border: 1px solid #0f3460;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e94560; }
    .metric-label { font-size: 0.85rem; color: #a0a0b0; margin-top: 4px; }
    .bottleneck-high { border-left: 4px solid #e94560; padding-left: 12px; }
    .bottleneck-medium { border-left: 4px solid #f59e0b; padding-left: 12px; }
    .bottleneck-info { border-left: 4px solid #3b82f6; padding-left: 12px; }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ───────────────────────────────────────────────────────────────

@st.cache_data
def load_results(results_dir="results"):
    """Load all benchmark JSON files from the results directory."""
    files = glob.glob(f"{results_dir}/benchmark_*.json")
    if not files:
        return None
    reports = []
    for f in files:
        with open(f) as fh:
            reports.append(json.load(fh))
    return reports


def load_demo_data():
    """Generate demo data so the dashboard works without running benchmarks."""
    import numpy as np

    operations = ["allreduce", "allgather", "broadcast"]
    sizes = [1, 4, 16, 64, 128, 256, 512]
    backends = ["gloo", "nccl"]

    collective_results = []
    for backend in backends:
        base_factor = 1.0 if backend == "gloo" else 0.6  # NCCL is faster
        for op in operations:
            op_factor = {"allreduce": 1.0, "allgather": 1.1, "broadcast": 0.7}[op]
            for size in sizes:
                latency = (0.5 + size * 0.08) * base_factor * op_factor + np.random.normal(0, 0.05)
                bw = (size * 8 / 1000) / (latency / 1000) if latency > 0 else 0
                collective_results.append({
                    "operation": op,
                    "size_mb": size,
                    "backend": backend,
                    "world_size": 4,
                    "mean_latency_ms": round(max(latency, 0.1), 3),
                    "std_latency_ms": round(abs(np.random.normal(0.1, 0.02)), 3),
                    "p50_latency_ms": round(max(latency * 0.95, 0.1), 3),
                    "p99_latency_ms": round(max(latency * 1.3, 0.1), 3),
                    "bandwidth_gbps": round(max(bw, 0.01), 3),
                })

    overlap_results = []
    for size in [1024, 2048, 4096]:
        compute = size * 0.002 + np.random.normal(0, 0.1)
        comm = size * 0.0035 + np.random.normal(0, 0.15)
        total = compute + comm
        overlap_results.append({
            "matrix_size": size,
            "compute_ms": round(max(compute, 0.1), 3),
            "communication_ms": round(max(comm, 0.1), 3),
            "total_ms": round(max(total, 0.2), 3),
            "comm_pct": round((comm / total) * 100, 2) if total > 0 else 0,
            "bottleneck": "network" if comm > compute else "compute",
        })

    return {
        "timestamp": "2025-03-05T10:00:00",
        "backend": "gloo",
        "system_info": {
            "hostname": "gpu-node-01",
            "gpu_count": 4,
            "ram_gb": 128,
            "cpu_count": 64,
            "gpu_names": [{"name": "NVIDIA A100-SXM4-80GB", "memory_gb": 80}] * 4,
            "nics": [{"interface": "eth0", "ip": "10.0.0.1"}, {"interface": "ib0", "ip": "10.0.1.1"}],
        },
        "network_baseline": {"ping_localhost_ms": 0.04, "estimated_loopback_bw_gbps": 45.2},
        "collective_benchmarks": collective_results,
        "compute_comm_overlap": overlap_results,
        "analysis": {
            "summary": {
                "allreduce": {"peak_bandwidth_gbps": 12.5, "optimal_message_size_mb": 256, "avg_latency_ms": 8.2},
                "allgather": {"peak_bandwidth_gbps": 10.8, "optimal_message_size_mb": 256, "avg_latency_ms": 9.1},
                "broadcast": {"peak_bandwidth_gbps": 15.2, "optimal_message_size_mb": 512, "avg_latency_ms": 5.6},
            },
            "bottlenecks": [
                {"type": "network_bound", "severity": "high",
                 "detail": "3/3 workload configs are network-bound. Communication overhead dominates compute time."},
                {"type": "small_message_latency", "severity": "medium",
                 "detail": "Average latency for small messages (<=4MB): 1.85ms. This can bottleneck gradient sync in models with many small layers."},
            ],
            "recommendations": [
                "Consider upgrading to higher-bandwidth interconnects (e.g., 200G/400G InfiniBand or RoCE).",
                "Enable gradient compression or mixed-precision allreduce to reduce message sizes.",
                "Investigate overlap of compute and communication using async allreduce.",
                "Consider gradient bucketing to batch small allreduce operations.",
            ],
        },
    }


# ─── Dashboard ──────────────────────────────────────────────────────────────────

def main():
    st.title("🔬 GPU Cluster Network Bottleneck Analyzer")
    st.caption("Profiling distributed AI training communication patterns")

    # Load data
    reports = load_results()
    if reports:
        report = reports[0]  # Use first report
        st.success(f"Loaded benchmark from {report['timestamp']}")
    else:
        st.info("No benchmark results found. Showing demo data. Run `python benchmark.py` to generate real results.")
        report = load_demo_data()

    sys_info = report["system_info"]
    analysis = report["analysis"]
    coll_df = pd.DataFrame(report["collective_benchmarks"])
    overlap_df = pd.DataFrame(report["compute_comm_overlap"])

    # ── System Overview ──
    st.header("System Overview")
    cols = st.columns(4)
    with cols[0]:
        st.metric("GPUs", sys_info["gpu_count"])
    with cols[1]:
        gpu_name = sys_info["gpu_names"][0]["name"] if sys_info["gpu_names"] else "N/A"
        st.metric("GPU Model", gpu_name)
    with cols[2]:
        st.metric("RAM", f"{sys_info['ram_gb']} GB")
    with cols[3]:
        st.metric("CPUs", sys_info["cpu_count"])

    st.divider()

    # ── Bottleneck Summary ──
    st.header("⚠️ Bottleneck Analysis")

    for bn in analysis["bottlenecks"]:
        severity = bn["severity"]
        icon = {"high": "🔴", "medium": "🟡", "info": "🔵"}[severity]
        css_class = f"bottleneck-{severity}"
        st.markdown(
            f'<div class="{css_class}"><strong>{icon} {bn["type"].replace("_", " ").title()}</strong>'
            f'<br>{bn["detail"]}</div><br>',
            unsafe_allow_html=True,
        )

    with st.expander("📋 Product Recommendations", expanded=True):
        for i, rec in enumerate(analysis["recommendations"], 1):
            st.markdown(f"**{i}.** {rec}")

    st.divider()

    # ── Bandwidth vs Message Size ──
    st.header("Bandwidth vs Message Size")

    fig_bw = px.line(
        coll_df,
        x="size_mb",
        y="bandwidth_gbps",
        color="operation",
        facet_col="backend" if coll_df["backend"].nunique() > 1 else None,
        markers=True,
        labels={"size_mb": "Message Size (MB)", "bandwidth_gbps": "Bandwidth (Gbps)", "operation": "Operation"},
        template="plotly_dark",
    )
    fig_bw.update_layout(height=400, xaxis_type="log")
    st.plotly_chart(fig_bw, use_container_width=True)

    # ── Latency vs Message Size ──
    st.header("Latency vs Message Size")

    fig_lat = px.line(
        coll_df,
        x="size_mb",
        y="mean_latency_ms",
        color="operation",
        facet_col="backend" if coll_df["backend"].nunique() > 1 else None,
        markers=True,
        labels={"size_mb": "Message Size (MB)", "mean_latency_ms": "Mean Latency (ms)", "operation": "Operation"},
        template="plotly_dark",
    )
    fig_lat.update_layout(height=400, xaxis_type="log")
    st.plotly_chart(fig_lat, use_container_width=True)

    # ── Latency Distribution (P50 vs P99) ──
    st.header("Latency Distribution: P50 vs P99")

    allreduce_df = coll_df[coll_df["operation"] == "allreduce"]
    fig_pct = go.Figure()
    for backend in allreduce_df["backend"].unique():
        bdf = allreduce_df[allreduce_df["backend"] == backend]
        fig_pct.add_trace(go.Bar(name=f"P50 ({backend})", x=bdf["size_mb"].astype(str), y=bdf["p50_latency_ms"]))
        fig_pct.add_trace(go.Bar(name=f"P99 ({backend})", x=bdf["size_mb"].astype(str), y=bdf["p99_latency_ms"]))
    fig_pct.update_layout(
        barmode="group", template="plotly_dark", height=400,
        xaxis_title="Message Size (MB)", yaxis_title="Latency (ms)",
    )
    st.plotly_chart(fig_pct, use_container_width=True)

    st.divider()

    # ── Compute vs Communication Breakdown ──
    st.header("Compute vs Communication Breakdown")

    fig_overlap = go.Figure()
    fig_overlap.add_trace(go.Bar(
        name="Compute",
        x=overlap_df["matrix_size"].astype(str),
        y=overlap_df["compute_ms"],
        marker_color="#3b82f6",
    ))
    fig_overlap.add_trace(go.Bar(
        name="Communication",
        x=overlap_df["matrix_size"].astype(str),
        y=overlap_df["communication_ms"],
        marker_color="#e94560",
    ))
    fig_overlap.update_layout(
        barmode="stack", template="plotly_dark", height=400,
        xaxis_title="Matrix Size (NxN)", yaxis_title="Time (ms)",
    )
    st.plotly_chart(fig_overlap, use_container_width=True)

    # Communication percentage table
    st.subheader("Communication Overhead %")
    styled_overlap = overlap_df[["matrix_size", "compute_ms", "communication_ms", "comm_pct", "bottleneck"]].copy()
    styled_overlap.columns = ["Matrix Size", "Compute (ms)", "Communication (ms)", "Comm %", "Bottleneck"]
    st.dataframe(styled_overlap, use_container_width=True, hide_index=True)

    st.divider()

    # ── Backend Comparison ──
    if coll_df["backend"].nunique() > 1:
        st.header("Backend Comparison: Gloo vs NCCL")

        allreduce_cmp = coll_df[coll_df["operation"] == "allreduce"]
        fig_cmp = px.bar(
            allreduce_cmp,
            x="size_mb",
            y="bandwidth_gbps",
            color="backend",
            barmode="group",
            labels={"size_mb": "Message Size (MB)", "bandwidth_gbps": "Bandwidth (Gbps)"},
            template="plotly_dark",
        )
        fig_cmp.update_layout(height=400, xaxis_type="log")
        st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Peak Performance Summary ──
    st.header("📊 Peak Performance Summary")
    summary_data = []
    for op, stats in analysis["summary"].items():
        summary_data.append({
            "Operation": op.title(),
            "Peak BW (Gbps)": stats["peak_bandwidth_gbps"],
            "Optimal Size (MB)": stats["optimal_message_size_mb"],
            "Avg Latency (ms)": stats["avg_latency_ms"],
        })
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # ── Raw Data Export ──
    with st.expander("📁 Export Raw Data"):
        st.download_button(
            "Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"network_benchmark_{report['timestamp'][:10]}.json",
            mime="application/json",
        )
        st.download_button(
            "Download Collective Benchmarks (CSV)",
            data=coll_df.to_csv(index=False),
            file_name="collective_benchmarks.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
