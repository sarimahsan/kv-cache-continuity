#!/usr/bin/env python3
"""
plot_cheap_pass.py -- Generates publication-ready figures and markdown tables
from cheap_pass_results JSON outputs.

Generates:
  1. fidelity_curves_E{E}.png: Per-layer Cosine Similarity & Relative MSE (NativeProj vs Adapter)
  2. kl_vs_depth.png: Test Split KL vs Exit Depth E (Oracle, NativeProj, Adapter) with ln(2) gate line
  3. capacity_sweep.png: KL & Top-1 Match vs Adapter Rank r
  4. cheap_pass_report.md: Comprehensive results table with 95% bootstrap CIs and gate decisions
"""

import argparse
import json
import math
import os
import numpy as np
import sys

LN2 = math.log(2.0)  # 0.6931


def generate_plots_and_tables(json_path: str, out_dir: str = None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print(" [Warning] matplotlib not installed; skipping graphic plot generation.")

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if out_dir is None:
        out_dir = os.path.dirname(json_path) or "cheap_pass_out"
    os.makedirs(out_dir, exist_ok=True)

    meta = data.get("meta", {})
    model_name = meta.get("model", "Model")
    gate_ln2 = meta.get("gate_ln2", LN2)

    # ------------------------------------------------------------------------
    # 1. Markdown Summary Table
    # ------------------------------------------------------------------------
    md_lines = []
    md_lines.append("# Experiment Report: Learned 'Cheap Pass' KV Adapters for Early Exit\n")
    md_lines.append(f"**Model:** `{model_name}` | **Pre-registered Gate:** $\\text{{KL}} < \\ln(2) \\approx {gate_ln2:.4f}$ nats at $E=18$\n")
    md_lines.append("## 1. Attention-Only Test Split KL & Fidelity\n")
    md_lines.append("| Exit Depth $E$ | Rel. Depth | Method | Forward KL (nats) [95% CI] | Top-1 Match (%) [95% CI] | Pre-Registered Gate |\n")
    md_lines.append("| :--- | :---: | :--- | :---: | :---: | :---: |\n")

    test_kl = data.get("test_attention_kl", {})
    layers_total = meta.get("layers", 36)

    for e_key, methods in test_kl.items():
        E_val = int(e_key.replace("E_", ""))
        rel_d = f"{100 * E_val / layers_total:.1f}%"
        for m_name in ["Oracle", "NativeProj", "Adapter", "ZeroInit"]:
            if m_name not in methods:
                continue
            m = methods[m_name]
            kl_str = f"{m['kl_mean']:.4f} [{m['kl_ci95'][0]:.4f}, {m['kl_ci95'][1]:.4f}]"
            top_str = f"{m['top1_mean']:.2f}% [{m['top1_ci95'][0]:.2f}, {m['top1_ci95'][1]:.2f}]"
            
            gate_status = "--"
            if m_name == "Adapter":
                if E_val == 18 or E_val == list(test_kl.keys())[0]:
                    gate_status = "**PASSED (Gate < 0.6931)**" if m['kl_mean'] < gate_ln2 else "**FAILED (Gate >= 0.6931)**"
                else:
                    nat_kl = methods.get("NativeProj", {}).get("kl_mean", float("inf"))
                    gate_status = "Improved vs Native" if m['kl_mean'] < nat_kl else "Regression vs Native"

            md_lines.append(f"| $E={E_val}$ | {rel_d} | **{m_name}** | {kl_str} | {top_str} | {gate_status} |\n")

    # Controls Summary
    controls = data.get("controls", {})
    if controls:
        md_lines.append("\n## 2. Experimental Controls & Robustness Checks\n")
        md_lines.append("| Control Experiment | Metric (KL) | Expected Outcome | Observed Result |\n")
        md_lines.append("| :--- | :---: | :--- | :--- |\n")
        
        # Zero init check
        zero_init = None
        for e_key, methods in test_kl.items():
            if "ZeroInit" in methods and "NativeProj" in methods:
                diff = abs(methods["ZeroInit"]["kl_mean"] - methods["NativeProj"]["kl_mean"])
                md_lines.append(f"| **Zero-Init Check** | Diff = {diff:.6f} | Exact match with NativeProj | {'Passed (Identical)' if diff < 1e-4 else 'Failed'} |\n")
                break
        
        # Shuffle control
        if "shuffle" in controls:
            sh_kl = controls["shuffle"]["kl_mean"]
            md_lines.append(f"| **Shuffle Control** | KL = {sh_kl:.4f} | Must degrade/fail | {'Passed (Failed gracefully)' if sh_kl > 0.5 else 'Suspicious'} |\n")
        
        # Sharing ablation
        if "sharing_ablation" in controls:
            sh_ab = controls["sharing_ablation"]
            md_lines.append(f"| **Sharing Ablation** | KL = {sh_ab['kl_mean']:.4f} | Test parameter-sharing penalty | Top-1: {sh_ab['top1_mean']:.2f}% |\n")

    report_path = os.path.join(out_dir, "cheap_pass_report.md")
    with open(report_path, "w") as f:
        f.writelines(md_lines)
    print(f" [Report] Wrote markdown table summary to: {report_path}")

    # ------------------------------------------------------------------------
    # 2. Plots (Matplotlib)
    # ------------------------------------------------------------------------
    if plt is None:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Plot A: KL vs Exit Depth E
    depths = []
    nat_kls, nat_los, nat_his = [], [], []
    ad_kls, ad_los, ad_his = [], [], []

    for e_key, methods in test_kl.items():
        E_val = int(e_key.replace("E_", ""))
        depths.append(E_val)
        if "NativeProj" in methods:
            m = methods["NativeProj"]
            nat_kls.append(m["kl_mean"])
            nat_los.append(m["kl_ci95"][0])
            nat_his.append(m["kl_ci95"][1])
        if "Adapter" in methods:
            m = methods["Adapter"]
            ad_kls.append(m["kl_mean"])
            ad_los.append(m["kl_ci95"][0])
            ad_his.append(m["kl_ci95"][1])

    if depths and ad_kls:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
        ax.axhline(gate_ln2, color="crimson", linestyle="--", linewidth=1.5, label=f"Pre-registered Gate: ln(2) = {gate_ln2:.4f} nats")
        
        if nat_kls:
            yerr_nat = [np.array(nat_kls) - np.array(nat_los), np.array(nat_his) - np.array(nat_kls)]
            ax.errorbar(depths, nat_kls, yerr=yerr_nat, fmt="o-", color="#e67e22", capsize=4, linewidth=2, label="NativeProj (Baseline)")
        
        yerr_ad = [np.array(ad_kls) - np.array(ad_los), np.array(ad_his) - np.array(ad_kls)]
        ax.errorbar(depths, ad_kls, yerr=yerr_ad, fmt="s-", color="#2980b9", capsize=4, linewidth=2, label="Cheap Pass Adapter (Variant A)")

        ax.set_xlabel("Exit Depth E (Layer)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Attention-Only Forward KL (nats)", fontsize=11, fontweight="bold")
        ax.set_title(f"Test Split KL Divergence vs Exit Depth ({model_name})", fontsize=12, fontweight="bold")
        ax.legend(frameon=True, fontsize=10)
        fig.tight_layout()
        plot_path = os.path.join(out_dir, "kl_vs_depth.png")
        fig.savefig(plot_path)
        plt.close(fig)
        print(f" [Plot] Saved: {plot_path}")

    # Plot B: Capacity Sweep vs Rank
    cap_sweep = controls.get("capacity_sweep", {})
    if cap_sweep:
        ranks = []
        r_kls = []
        r_tops = []
        for r_key, met in cap_sweep.items():
            r_val = int(r_key.replace("r_", ""))
            ranks.append(r_val)
            r_kls.append(met["kl_mean"])
            r_tops.append(met["top1_mean"])

        fig, ax1 = plt.subplots(figsize=(6.5, 4), dpi=300)
        ax2 = ax1.twinx()

        p1 = ax1.plot(ranks, r_kls, "o-", color="#2980b9", linewidth=2, label="Forward KL (nats)")
        p2 = ax2.plot(ranks, r_tops, "s--", color="#27ae60", linewidth=2, label="Top-1 Agreement (%)")

        ax1.axhline(gate_ln2, color="crimson", linestyle=":", label="ln(2) Gate")
        ax1.set_xlabel("Adapter Bottleneck Rank (r)", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Forward KL (nats)", color="#2980b9", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Top-1 Match (%)", color="#27ae60", fontsize=11, fontweight="bold")
        ax1.set_title("Capacity Sweep: Performance vs Rank", fontsize=12, fontweight="bold")
        
        lines = p1 + p2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right", frameon=True)
        fig.tight_layout()
        plot_cap = os.path.join(out_dir, "capacity_sweep.png")
        fig.savefig(plot_cap)
        plt.close(fig)
        print(f" [Plot] Saved: {plot_cap}")

    # Plot C: Per-Layer Fidelity Curves (Cosine & Relative MSE vs Layer)
    fidelity_gate = data.get("fidelity_gate", {})
    for e_key, gate_info in fidelity_gate.items():
        E_val = int(e_key.replace("E_", ""))
        details = gate_info.get("details", {})
        if not details:
            continue
        
        layers = []
        nat_ck, ad_ck = [], []
        nat_cv, ad_cv = [], []
        nat_mv, ad_mv = [], []

        for l_key, l_data in details.items():
            l_num = int(l_key.replace("layer_", ""))
            layers.append(l_num)
            nat_ck.append(l_data["native"]["cos_k"])
            ad_ck.append(l_data["adapter"]["cos_k"])
            nat_cv.append(l_data["native"]["cos_v"])
            ad_cv.append(l_data["adapter"]["cos_v"])
            nat_mv.append(l_data["native"]["rel_mse_v"])
            ad_mv.append(l_data["adapter"]["rel_mse_v"])

        if layers:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
            
            # Subplot 1: Cosine Similarity
            ax1.plot(layers, nat_ck, "o--", color="#f39c12", label="Native Key Cos")
            ax1.plot(layers, ad_ck, "o-", color="#e67e22", label="Adapter Key Cos")
            ax1.plot(layers, nat_cv, "s--", color="#9b59b6", label="Native Value Cos")
            ax1.plot(layers, ad_cv, "s-", color="#8e44ad", label="Adapter Value Cos")
            ax1.set_xlabel("Layer Index (l)", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Cosine Similarity vs Oracle", fontsize=11, fontweight="bold")
            ax1.set_title(f"Per-Layer Cosine Fidelity (E={E_val})", fontsize=12, fontweight="bold")
            ax1.legend(frameon=True, fontsize=9)
            ax1.set_ylim(-0.05, 1.05)

            # Subplot 2: Relative MSE
            ax2.plot(layers, nat_mv, "s--", color="#e74c3c", label="Native Value Rel-MSE")
            ax2.plot(layers, ad_mv, "s-", color="#27ae60", label="Adapter Value Rel-MSE")
            ax2.set_xlabel("Layer Index (l)", fontsize=11, fontweight="bold")
            ax2.set_ylabel("Relative MSE ||V_hat - V||^2 / ||V||^2", fontsize=11, fontweight="bold")
            ax2.set_title(f"Per-Layer Value Error (E={E_val})", fontsize=12, fontweight="bold")
            ax2.legend(frameon=True, fontsize=9)

            fig.tight_layout()
            fid_plot = os.path.join(out_dir, f"fidelity_curves_E{E_val}.png")
            fig.savefig(fid_plot)
            plt.close(fig)
            print(f" [Plot] Saved: {fid_plot}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json", default="cheap_pass_out/cheap_pass_results_Qwen_Qwen2.5-3B.json", help="Results JSON file")
    p.add_argument("--out_dir", default=None, help="Output directory for plots and markdown")
    args = p.parse_args()
    generate_plots_and_tables(args.json, args.out_dir)
