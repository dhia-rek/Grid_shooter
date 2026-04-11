"""
plot.py — Generate training figures matching PDF 3 slide 18.

Run AFTER train.py:
    python plot.py

Produces:  outputs/training_curves.png
           (4 subplots the prof explicitly asks for in the lab)

PDF 3 slide 18 asks to track:
  1. Episode return (raw)              → top-left
  2. Moving average return (last 50)  → top-right
  3. Win rate over time               → bottom-left
  4. Episode length over time         → bottom-right (bonus)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── load saved history ────────────────────────────────────────────────────────
try:
    with open("outputs/training_history.json") as f:
        h = json.load(f)
except FileNotFoundError:
    print("ERROR: outputs/training_history.json not found.")
    print("Run  python train.py  first, then come back here.")
    raise

episodes    = h["episode"]
raw_returns = h["return"]
avg50       = h["avg50"]
win_rates   = h["win_rate_pct"]

# ── figure setup ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8))
fig.suptitle(
    "REINFORCE on Grid Shooter  —  Training Curves\n"
    "(Follows PDF 3, slide 18 evaluation metrics)",
    fontsize=13, fontweight="bold", y=0.98
)

gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32)

# colour scheme
RAW_COL  = "#4e79a7"   # blue  – raw episode return
AVG_COL  = "#f28e2b"   # orange – moving average
WIN_COL  = "#59a14f"   # green – win rate
ZERO_COL = "#d3d3d3"   # grey  – reference line at 0


def _style(ax, xlabel, ylabel, title, ref=0.0):
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(ref, color=ZERO_COL, linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)


# ── subplot 1: raw episode return ─────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(episodes, raw_returns, color=RAW_COL, alpha=0.7,
         linewidth=1.0, label="Episode return")
ax1.scatter(episodes, raw_returns, color=RAW_COL, s=12, alpha=0.5)
_style(ax1,
       xlabel="Episode",
       ylabel="Total return",
       title="1. Raw Episode Return\n(spiky — expected for REINFORCE)")

# Annotate best episode
best_idx = int(np.argmax(raw_returns))
ax1.annotate(
    f"best: {raw_returns[best_idx]:+.1f}",
    xy=(episodes[best_idx], raw_returns[best_idx]),
    xytext=(0, 12), textcoords="offset points",
    fontsize=7, color=RAW_COL, arrowprops=dict(arrowstyle="-", color=RAW_COL)
)

# ── subplot 2: moving average (last 50) ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(episodes, raw_returns, color=RAW_COL, alpha=0.25,
         linewidth=0.8, label="Raw return")
ax2.plot(episodes, avg50, color=AVG_COL, linewidth=2.0,
         label="Avg last 50 eps")
_style(ax2,
       xlabel="Episode",
       ylabel="Return",
       title="2. Moving Average (last 50 eps)\n(PDF 3 slide 18 — the trend line)")

# ── subplot 3: win rate ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(episodes, win_rates, color=WIN_COL, linewidth=2.0,
         label="Win rate %")
ax3.fill_between(episodes, win_rates, alpha=0.15, color=WIN_COL)
ax3.set_ylim(0, 100)
ax3.axhline(50, color="grey", linewidth=0.7, linestyle=":",
            label="50% reference")
_style(ax3,
       xlabel="Episode",
       ylabel="Win rate (%)",
       title="3. Win Rate Over Time\n(% episodes ending in agent killing the enemy)",
       ref=0)
ax3.legend(fontsize=8)   # re-draw legend after extra line

# ── subplot 4: improvement summary (bar chart) ────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
if len(episodes) >= 4:
    # Compare first quarter vs last quarter average return
    q = len(raw_returns) // 4
    early_avg = float(np.mean(raw_returns[:q]))
    late_avg  = float(np.mean(raw_returns[-q:]))

    bars = ax4.bar(
        ["Early training\n(first 25%)", "Late training\n(last 25%)"],
        [early_avg, late_avg],
        color=[RAW_COL, AVG_COL], edgecolor="white", width=0.5
    )
    for bar, val in zip(bars, [early_avg, late_avg]):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 val + (abs(val) * 0.04) * (1 if val >= 0 else -1),
                 f"{val:+.2f}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold")

    ax4.axhline(0, color=ZERO_COL, linewidth=0.8, linestyle="--")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.set_ylabel("Average return", fontsize=9)
    ax4.set_title("4. Policy Improvement Summary\n"
                  "(early vs late average return)", fontsize=10,
                  fontweight="bold")

# ── save ──────────────────────────────────────────────────────────────────────
import os; os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
print("Saved: outputs/training_curves.png")
print()
print("What each plot shows (for your exam presentation):")
print("  Plot 1: Raw returns — spiky because REINFORCE is high-variance.")
print("          Normal. Use the moving average to see the true trend.")
print("  Plot 2: Moving average — this is the 'solved?' indicator.")
print("          PDF 3 asks: did the moving average stabilise above 0?")
print("  Plot 3: Win rate — direct measure of game performance.")
print("          Should climb as the policy learns to kill the enemy.")
print("  Plot 4: Early vs late return — shows the policy gradient worked.")
print("          If late > early, REINFORCE improved the policy. QED.")
