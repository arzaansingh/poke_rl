"""
Generate MDP diagram as a high-res PNG using matplotlib.
Pokeball-shaped Agent, clean Environment box, symmetric curved arrows.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Wedge
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(-0.5, 8.5)
ax.set_aspect('equal')
ax.axis('off')

# Colors
TULANE_GREEN = '#006747'
TULANE_BLUE = '#418FDE'
TULANE_DARK = '#003D2B'
ORANGE = '#E65100'
POKEBALL_RED = '#CC0000'

# ═══════════════════════════════════════
#  POKEBALL (Agent) — centered at (3.2, 4)
# ═══════════════════════════════════════
ball_cx, ball_cy, ball_r = 3.2, 4.0, 1.7

top_half = Wedge((ball_cx, ball_cy), ball_r, 0, 180, facecolor=POKEBALL_RED, edgecolor='#222', linewidth=3, zorder=3)
ax.add_patch(top_half)
bot_half = Wedge((ball_cx, ball_cy), ball_r, 180, 360, facecolor='white', edgecolor='#222', linewidth=3, zorder=3)
ax.add_patch(bot_half)
ax.plot([ball_cx - ball_r, ball_cx + ball_r], [ball_cy, ball_cy],
        color='#222', linewidth=6, zorder=4, solid_capstyle='butt')
ax.add_patch(plt.Circle((ball_cx, ball_cy), 0.38, facecolor='white', edgecolor='#222', linewidth=3.5, zorder=5))
ax.add_patch(plt.Circle((ball_cx, ball_cy), 0.16, facecolor='white', edgecolor='#222', linewidth=2.5, zorder=6))

# Agent label
ax.text(ball_cx, ball_cy + ball_r + 0.6, 'Agent', fontsize=28, fontweight='bold',
        ha='center', va='bottom', color='#222', zorder=10)

# Text inside pokeball — BIGGER
ax.text(ball_cx, ball_cy + 0.95, 'Tabular Q-Learning', fontsize=17, fontweight='bold',
        ha='center', va='center', color='white', zorder=7)
ax.text(ball_cx, ball_cy + 0.5, 'Policy', fontsize=15, fontweight='bold',
        ha='center', va='center', color='#ffffffdd', zorder=7)
ax.text(ball_cx, ball_cy - 0.6, r'$\pi(s) = \arg\max_a Q(s,a)$', fontsize=16, fontweight='bold',
        ha='center', va='center', color='#222', zorder=7)
ax.text(ball_cx, ball_cy - 1.1, r'Watkins $Q(\lambda)$', fontsize=15, fontweight='bold',
        ha='center', va='center', color='#555', zorder=7)

# ═══════════════════════════════════════
#  ENVIRONMENT BOX — centered at (12.2, 4)
# ═══════════════════════════════════════
env_cx, env_cy = 12.2, 4.0
env_w, env_h = 4.8, 3.4
env_x, env_y = env_cx - env_w / 2, env_cy - env_h / 2

env_box = FancyBboxPatch((env_x, env_y), env_w, env_h,
                          boxstyle="round,pad=0.15", facecolor=TULANE_BLUE,
                          edgecolor='#2a6cb8', linewidth=2.5, zorder=3)
ax.add_patch(env_box)

ax.text(env_cx, env_y + env_h + 0.6, 'Environment', fontsize=28, fontweight='bold',
        ha='center', va='bottom', color='#222', zorder=10)
ax.text(env_cx, env_cy + 1.0, 'Pokémon Showdown', fontsize=21, fontweight='bold',
        ha='center', va='center', color='white', zorder=7)
ax.text(env_cx, env_cy + 0.4, 'Server', fontsize=19, fontweight='bold',
        ha='center', va='center', color='white', zorder=7)
ax.plot([env_x + 0.5, env_x + env_w - 0.5], [env_cy - 0.1, env_cy - 0.1],
        color='#ffffff55', linewidth=1.5, zorder=5)
ax.text(env_cx, env_cy - 0.5, 'Gen 4 OU Random Battles', fontsize=16, fontweight='bold',
        ha='center', va='center', color='white', zorder=7)
ax.text(env_cx, env_cy - 1.1, r"$P(s' \mid s, a)$ — stochastic", fontsize=15,
        ha='center', va='center', color='#ffffffee', zorder=7, fontstyle='italic')
ax.text(env_cx, env_cy - 1.5, 'partially observable', fontsize=13, fontweight='semibold',
        ha='center', va='center', color='#ffffffaa', zorder=7)

# ═══════════════════════════════════════
#  HELPER: compute midpoint of arc3 bezier
# ═══════════════════════════════════════
def arc3_midpoint(p0, p1, rad):
    """Compute the midpoint (t=0.5) of a quadratic bezier created by arc3.

    Matches matplotlib's actual Arc3 control point formula:
        cx = (x1+x2)/2 + rad * (y2-y1)
        cy = (y1+y2)/2 - rad * (x2-x1)
    """
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    # matplotlib arc3 control point (no chord_len multiplier!)
    cp = np.array([(p0[0] + p1[0]) / 2 + rad * dy,
                   (p0[1] + p1[1]) / 2 - rad * dx])
    # Quadratic bezier at t=0.5: B(0.5) = 0.25*p0 + 0.5*cp + 0.25*p1
    mid = 0.25 * p0 + 0.5 * cp + 0.25 * p1
    return mid

# ═══════════════════════════════════════
#  ARROWS
# ═══════════════════════════════════════

# midpoint for labels on the straight action arrow
amid_x = (ball_cx + ball_r + env_x) / 2

# --- STATE (curved arc above, Environment → Agent) ---
state_start = (env_cx, env_y + env_h + 0.15)
state_end = (ball_cx, ball_cy + ball_r + 0.3)
state_rad = 0.4

state_arrow = FancyArrowPatch(
    state_start, state_end,
    connectionstyle=f"arc3,rad={state_rad}",
    arrowstyle='->', mutation_scale=30,
    color=TULANE_BLUE, linewidth=3.5, zorder=2
)
ax.add_patch(state_arrow)

smid = arc3_midpoint(state_start, state_end, state_rad)
state_box = FancyBboxPatch((amid_x - 1.15, smid[1] - 0.32), 2.3, 0.64,
                            boxstyle="round,pad=0.08", facecolor='white',
                            edgecolor=TULANE_BLUE, linewidth=2, zorder=8)
ax.add_patch(state_box)
ax.text(amid_x, smid[1], r'State  $s_{t+1}$', fontsize=18, fontweight='bold',
        ha='center', va='center', color=TULANE_BLUE, zorder=9)
ax.text(amid_x, smid[1] + 0.55, 'Discretized battle state', fontsize=15,
        fontweight='bold', ha='center', va='center', color='#333', zorder=9)

# --- ACTION (straight, Agent → Environment) ---
action_arrow = FancyArrowPatch(
    (ball_cx + ball_r + 0.2, ball_cy),
    (env_x - 0.2, env_cy),
    arrowstyle='->', mutation_scale=30,
    color=TULANE_GREEN, linewidth=3.5, zorder=2
)
ax.add_patch(action_arrow)

action_box = FancyBboxPatch((amid_x - 1.15, ball_cy - 0.32), 2.3, 0.64,
                             boxstyle="round,pad=0.08", facecolor='white',
                             edgecolor=TULANE_GREEN, linewidth=2, zorder=8)
ax.add_patch(action_box)
ax.text(amid_x, ball_cy, r'Action  $a_t$', fontsize=18, fontweight='bold',
        ha='center', va='center', color=TULANE_DARK, zorder=9)
ax.text(amid_x, ball_cy + 0.55, 'Move 1–4  |  Switch', fontsize=15,
        fontweight='bold', ha='center', va='center', color='#333', zorder=9)

# --- REWARD (curved arc below, Environment → Agent) ---
reward_start = (env_cx, env_y - 0.15)
reward_end = (ball_cx, ball_cy - ball_r - 0.3)
reward_rad = -0.4

reward_arrow = FancyArrowPatch(
    reward_start, reward_end,
    connectionstyle=f"arc3,rad={reward_rad}",
    arrowstyle='->', mutation_scale=30,
    color=ORANGE, linewidth=3.5, zorder=2
)
ax.add_patch(reward_arrow)

rmid = arc3_midpoint(reward_start, reward_end, reward_rad)
reward_box = FancyBboxPatch((amid_x - 1.25, rmid[1] - 0.32), 2.5, 0.64,
                             boxstyle="round,pad=0.08", facecolor='white',
                             edgecolor=ORANGE, linewidth=2, zorder=8)
ax.add_patch(reward_box)
ax.text(amid_x, rmid[1], r'Reward  $r_{t+1}$', fontsize=18, fontweight='bold',
        ha='center', va='center', color=ORANGE, zorder=9)
ax.text(amid_x, rmid[1] - 0.55, '±1 terminal + dense shaping', fontsize=15,
        fontweight='bold', ha='center', va='center', color='#333', zorder=9)

# ═══════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════
ax.text(8.0, 7.7, 'Markov Decision Process', fontsize=30, fontweight='bold',
        ha='center', va='center', color=TULANE_GREEN, zorder=10)

plt.tight_layout(pad=0.2)
out = '/Users/Arzaan/Documents/Tulane University/Courses/2026 Spring/CMPS 4020/PokeAgent/Research Experiments V5 Initialization Study/poster/figures/mdp_diagram.png'
plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
plt.close()
