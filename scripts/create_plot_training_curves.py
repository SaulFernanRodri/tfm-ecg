import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

epochs = list(range(1, 29))

auc_train = [0.8178,0.8784,0.8960,0.9049,0.9130,0.9147,0.9179,0.9193,0.9227,0.9247,
             0.9272,0.9292,0.9310,0.9324,0.9336,0.9359,0.9371,0.9453,0.9461,0.9487,
             0.9500,0.9524,0.9520,0.9576,0.9606,0.9618,0.9621,0.9642]

auc_val   = [0.8920,0.9005,0.9092,0.9159,0.9174,0.9220,0.9051,0.9227,0.9186,0.9197,
             0.9233,0.9234,0.9241,0.9237,0.9239,0.9194,0.9185,0.9286,0.9238,0.9195,
             0.9219,0.9262,0.9218,0.9199,0.9194,0.9192,0.9177,0.9145]

loss_train = [0.2681,0.2188,0.2032,0.1954,0.1885,0.1854,0.1826,0.1799,0.1770,0.1746,
              0.1712,0.1692,0.1667,0.1655,0.1641,0.1601,0.1589,0.1480,0.1457,0.1421,
              0.1406,0.1355,0.1357,0.1256,0.1222,0.1197,0.1183,0.1152]

loss_val   = [0.1268,0.1263,0.1205,0.1222,0.1113,0.1112,0.1241,0.1048,0.1110,0.1095,
              0.1090,0.1039,0.1129,0.1070,0.1081,0.1081,0.1099,0.1008,0.1056,0.1093,
              0.1077,0.1062,0.1051,0.1154,0.1142,0.1170,0.1191,0.1267]

BEST_EPOCH = 18  # época 18 (índice 17), la de mayor val_auc

BLUE  = "#2171B5"
GREEN = "#238B45"
RED   = "#CB181D"
GRID  = "#E8E8E8"

fig = plt.figure(figsize=(12, 8))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(2, 1, hspace=0.45)

# ── AUC ──────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(epochs, auc_train, color=BLUE, linewidth=2,   label="Entrenamiento")
ax1.plot(epochs, auc_val,   color=BLUE, linewidth=2,   label="Validación", linestyle="--")
ax1.axvline(x=BEST_EPOCH, color=RED, linewidth=1.2, linestyle=":", label=f"Mejor época ({BEST_EPOCH})")
ax1.annotate(f"val AUC = {max(auc_val):.4f}",
             xy=(BEST_EPOCH, max(auc_val)),
             xytext=(BEST_EPOCH + 1.5, max(auc_val) - 0.008),
             fontsize=9, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
ax1.set_ylabel("AUC macro", fontsize=12)
ax1.set_title("Curvas de entrenamiento — AUC macro", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlim(1, 28)
ax1.set_ylim(0.86, 0.97)
ax1.set_xticks(range(1, 29, 2))
ax1.grid(True, color=GRID, linewidth=0.8)
ax1.set_facecolor("white")
ax1.legend(fontsize=10, framealpha=0.9, edgecolor=GRID)
ax1.spines[["top", "right"]].set_visible(False)

# ── LOSS ─────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])
ax2.plot(epochs, loss_train, color=GREEN, linewidth=2, label="Entrenamiento")
ax2.plot(epochs, loss_val,   color=GREEN, linewidth=2, label="Validación", linestyle="--")
ax2.axvline(x=BEST_EPOCH, color=RED, linewidth=1.2, linestyle=":", label=f"Mejor época ({BEST_EPOCH})")
ax2.annotate(f"val Loss = {loss_val[BEST_EPOCH - 1]:.4f}",
             xy=(BEST_EPOCH, loss_val[BEST_EPOCH - 1]),
             xytext=(BEST_EPOCH + 1.5, loss_val[BEST_EPOCH - 1] + 0.008),
             fontsize=9, color=RED,
             arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
ax2.set_xlabel("Época", fontsize=12)
ax2.set_ylabel("Pérdida (ASL)", fontsize=12)
ax2.set_title("Curvas de entrenamiento — Pérdida (ASL)", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlim(1, 28)
ax2.set_ylim(0.08, 0.30)
ax2.set_xticks(range(1, 29, 2))
ax2.grid(True, color=GRID, linewidth=0.8)
ax2.set_facecolor("white")
ax2.legend(fontsize=10, framealpha=0.9, edgecolor=GRID)
ax2.spines[["top", "right"]].set_visible(False)

plt.savefig("curvas_entrenamiento_v5.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.show()
print("Figura guardada como curvas_entrenamiento_v5.png")