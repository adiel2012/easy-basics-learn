"""Regenerates the activation and derivative plots used in activation_functions.md."""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

z = np.linspace(-4, 4, 801)
erf = np.vectorize(math.erf)
Phi = 0.5 * (1 + erf(z / math.sqrt(2)))
phi = np.exp(-z**2 / 2) / math.sqrt(2 * math.pi)
sig = 1 / (1 + np.exp(-z))
th = np.tanh(z)

curves = {
    "ReLU": (np.maximum(0, z), (z > 0).astype(float)),
    "Sigmoid": (sig, sig * (1 - sig)),
    "Tanh": (th, 1 - th**2),
    "GELU": (z * Phi, Phi + z * phi),
}
colors = {"ReLU": "#1f77b4", "Sigmoid": "#ff7f0e", "Tanh": "#2ca02c", "GELU": "#d62728"}
out = Path(__file__).parent


def plot(idx, title, ylabel, ylim, fname):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for name, vals in curves.items():
        y = vals[idx].copy()
        if name == "ReLU" and idx == 1:          # don't draw the jump at 0 as a vertical line
            y[np.argmin(np.abs(z))] = np.nan
        ax.plot(z, y, label=name, color=colors[name], linewidth=2)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlim(-4, 4)
    ax.set_ylim(*ylim)
    ax.set_xlabel("z")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out / fname, dpi=150)
    plt.close(fig)


plot(0, "Activation functions f(z)", "f(z)", (-1.5, 4), "activation_curves.png")
plot(1, "Their derivatives f'(z): how much gradient passes back", "f'(z)", (-0.3, 1.3), "activation_derivatives.png")
print("saved")


# ReLU vs GELU vs SiLU near zero (used in §7)
z3 = np.linspace(-3, 3, 601)
Phi3 = 0.5 * (1 + erf(z3 / math.sqrt(2)))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(z3, np.maximum(0, z3), label="ReLU", color=colors["ReLU"], linewidth=2)
ax.plot(z3, z3 * Phi3, label="GELU", color=colors["GELU"], linewidth=2)
ax.plot(z3, z3 / (1 + np.exp(-z3)), label="SiLU / Swish", color="#9467bd", linewidth=2)
for name, xm, ym, col in [("GELU min ≈ −0.17", -0.75, -0.17, colors["GELU"]),
                          ("SiLU min ≈ −0.28", -1.28, -0.28, "#9467bd")]:
    ax.plot(xm, ym, "o", color=col)
    ax.annotate(name, (xm, ym), textcoords="offset points", xytext=(-10, -18), ha="right", color=col)
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_xlim(-3, 3)
ax.set_ylim(-0.6, 3)
ax.set_xlabel("z")
ax.set_ylabel("f(z)")
ax.set_title("ReLU vs. GELU vs. SiLU: same far away, different near 0")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(out / "relu_gelu_silu.png", dpi=150)
plt.close(fig)
print("saved comparison")
