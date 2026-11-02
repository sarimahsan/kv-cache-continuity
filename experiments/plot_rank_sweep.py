import matplotlib.pyplot as plt
import numpy as np
import os

# Data from results_v3.json (Qwen2.5-3B, L=18 rank sweep)
ranks = [16, 64, 256, 1024]
params_m = np.array([184624, 726208, 2892544, 11557888]) / 1e6
wiki_kl = [0.1695, 0.1609, 0.1552, 0.1514]
wiki_top1 = [79.32, 80.03, 80.18, 80.68]
c4_kl = [0.1545, 0.1529, 0.1493, 0.1444]
c4_top1 = [80.18, 80.30, 80.26, 80.41]

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

# ----------------- Panel 1: KL Divergence (Lower is better) -----------------
ax1.plot(ranks, wiki_kl, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='WikiText-2 (In-Domain)')
ax1.plot(ranks, c4_kl, 's--', color='#ff7f0e', linewidth=2, markersize=6, label='C4 (Held-Out Web)')

ax1.set_xscale('log', base=2)
ax1.set_xticks(ranks)
ax1.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax1.set_xlabel('Adapter Rank (Log Scale)', fontsize=9, fontweight='bold')
ax1.set_ylabel('KL Divergence (nats)', fontsize=9, fontweight='bold')
ax1.set_title('KL Divergence vs Capacity\n(Asymptotic Flat Floor)', fontsize=10, fontweight='bold')
ax1.set_ylim(0.12, 0.19)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=8, loc='upper right')

# Annotation on Panel 1
ax1.annotate('62.6× Params\nΔKL = -0.010 nats', 
             xy=(1024, 0.1444), xytext=(128, 0.130),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1),
             fontsize=7.5, fontweight='bold', bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffcc', alpha=0.8))

# ----------------- Panel 2: Top-1 Agreement (Higher is better) -----------------
ax2.plot(ranks, wiki_top1, 'o-', color='#1f77b4', linewidth=2, markersize=6, label='WikiText-2')
ax2.plot(ranks, c4_top1, 's--', color='#ff7f0e', linewidth=2, markersize=6, label='C4')

ax2.set_xscale('log', base=2)
ax2.set_xticks(ranks)
ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax2.set_xlabel('Adapter Rank (Log Scale)', fontsize=9, fontweight='bold')
ax2.set_ylabel('Top-1 Agreement (%)', fontsize=9, fontweight='bold')
ax2.set_title('Top-1 Agreement vs Capacity\n(Dead Flat on C4: +0.23%)', fontsize=10, fontweight='bold')
ax2.set_ylim(78.5, 82.0)
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize=8, loc='lower right')

# Annotate parameter range on top
for ax in (ax1, ax2):
    ax.tick_params(axis='both', which='major', labelsize=8)

plt.tight_layout()
out_dir = r"d:\CUDA-Graph\kv_continuity\New folder"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "rank_sweep_capacity_vs_info.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Graph saved successfully to: {out_path}")
