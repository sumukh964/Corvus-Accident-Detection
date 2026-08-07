# Generates arch_diagram.png -- CORVUS-themed architecture + workflow diagram
# for the presentation. Red/black HUD aesthetic matching the live dashboard.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BG      = "#0A0A0D"
PANEL   = "#12181F"
RED     = "#E4232B"
RED_DIM = "#7A1418"
AMBER   = "#FF8C1A"
WHITE   = "#F2F2F2"
GREY    = "#98A2AC"
GREEN   = "#3ADB76"
MONO    = "DejaVu Sans Mono"

fig, ax = plt.subplots(figsize=(13.0, 6.0), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 130)
ax.set_ylim(0, 60)
ax.axis("off")

def box(x, y, w, h, label, sub=None, edge=RED, face=PANEL, lw=2.2, fontsize=11, subcolor=GREY):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=1.6",
                        linewidth=lw, edgecolor=edge, facecolor=face, zorder=3)
    ax.add_patch(b)
    if sub:
        ax.text(x + w/2, y + h*0.62, label, ha="center", va="center",
                 color=WHITE, fontsize=fontsize, fontweight="bold", family=MONO, zorder=4)
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center",
                 color=subcolor, fontsize=fontsize-2.5, family=MONO, zorder=4)
    else:
        ax.text(x + w/2, y + h/2, label, ha="center", va="center",
                 color=WHITE, fontsize=fontsize, fontweight="bold", family=MONO, zorder=4)
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, color=RED, lw=2.2, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                         linewidth=lw, color=color, zorder=2, shrinkA=2, shrinkB=2)
    ax.add_patch(a)

# ---- main pipeline (left -> right, top row) ----
row_y = 42
w1, h1 = 16, 9
xs = [4, 24, 44, 64, 84, 104]
labels = [
    ("VIDEO INPUT", "CCTV / traffic cam"),
    ("YOLOv8-SEG", "detect + segment"),
    ("DEEPSORT", "track + Re-ID"),
    ("SIGNAL FUSION", "overlap/speed/contact"),
    ("SEVERITY SCORE", "weighted vote"),
    ("ACCIDENT EVENT", "confirmed + logged"),
]
boxes = []
for x, (lab, sub) in zip(xs, labels):
    edge = RED if lab != "ACCIDENT EVENT" else AMBER
    boxes.append(box(x, row_y, w1, h1, lab, sub, edge=edge))

for i in range(len(xs) - 1):
    x1 = xs[i] + w1
    x2 = xs[i+1]
    arrow(x1, row_y + h1/2, x2, row_y + h1/2)

# ---- zone monitoring branch (parallel input into signal fusion) ----
zone_x, zone_y, zone_w, zone_h = 44, 27, 16, 8
box(zone_x, zone_y, zone_w, zone_h, "ZONE MONITORING", "Intersection/Crosswalk/No-Entry", edge=GREEN, fontsize=10)
arrow(zone_x + zone_w/2, zone_y + zone_h, xs[3] + w1/2, row_y, color=GREEN)

# ---- output branch (from ACCIDENT EVENT fanning down to alert + log) ----
out_y = 27
out_w, out_h = 18, 8
out_x1, out_x2, out_x3 = 92, 112, 72
box(out_x3, out_y, out_w, out_h, "INCIDENT REPLAY", "clip + JSONL log", edge=AMBER, fontsize=10)
box(out_x1, out_y, out_w, out_h, "DISCORD ALERT", "webhook + screenshot", edge=AMBER, fontsize=10)
box(out_x2, out_y, out_w, out_h, "TELEGRAM ALERT", "bot API + screenshot", edge=AMBER, fontsize=10)

src_x = xs[5] + w1/2
src_y = row_y
for tx in (out_x3 + out_w/2, out_x1 + out_w/2, out_x2 + out_w/2):
    arrow(src_x, src_y, tx, out_y + out_h, color=AMBER)

# ---- HUD / live monitoring output ----
hud_x, hud_y, hud_w, hud_h = 92, 14, 32, 7
box(hud_x, hud_y, hud_w, hud_h, "LIVE HUD DASHBOARD", "threat gauge + incident log", edge=RED, fontsize=10)
arrow(out_x1 + out_w/2, out_y, hud_x + hud_w/2, hud_y + hud_h, color=RED)
arrow(out_x3 + out_w/2, out_y, hud_x + hud_w/2, hud_y + hud_h, color=RED)

# ---- bottom labels: module count / stack ----
ax.text(4, 19, "NINE COOPERATING MODULES", color=RED, fontsize=11, fontweight="bold", family=MONO)
mods = ["detector.py", "tracker.py", "zones.py", "behavior.py", "accident_logic.py",
        "scene_cut.py", "replay.py / logger.py", "alert.py", "utils.py (HUD)"]
for i, m in enumerate(mods):
    col = i % 3
    rowi = i // 3
    ax.text(4 + col*24, 15.5 - rowi*3.0, "▸ " + m, color=GREY, fontsize=9, family=MONO)

ax.text(4, 1.8, "main.py  —  orchestrator   |   config.py / edge_config.py  —  shared configuration & deployment switch",
        color=GREY, fontsize=9.5, family=MONO)

plt.tight_layout(pad=0.4)
plt.savefig("arch_diagram.png", facecolor=BG, dpi=200)
print("saved arch_diagram.png")
