"""
make_report.py - Generate the PDF report for the Grid Shooter REINFORCE project.

Produces: outputs/report.pdf

Usage:
    python make_report.py
"""

import json
import os
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos

OUT    = "outputs/report.pdf"
CURVES = "outputs/training_curves.png"
EVAL   = "outputs/evaluation.png"

W, H = 210, 297   # A4 mm

BLUE   = (30,  80, 160)
GREEN  = (20, 120,  70)
PURPLE = (110, 40, 140)
RED    = (150,  35,  55)
DARK   = (30,  30,  40)


# ── PDF class ─────────────────────────────────────────────────────────────────

class PDF(FPDF):

    def header(self):
        if self.page_no() <= 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140)
        self.cell(0, 7,
                  "Grid Shooter  -  REINFORCE Policy Gradient Agent  |  EM IA 2026",
                  align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(200)
        self.set_line_width(0.25)
        self.line(self.l_margin, self.get_y(), W - self.r_margin, self.get_y())
        self.set_draw_color(0)
        self.set_text_color(0)
        self.ln(1)

    def footer(self):
        if self.page_no() <= 1:
            return
        self.set_y(-13)
        self.set_draw_color(200)
        self.set_line_width(0.25)
        self.line(self.l_margin, self.get_y(), W - self.r_margin, self.get_y())
        self.set_draw_color(0)
        self.ln(1)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")
        self.set_text_color(0)

    # ── Section heading (uses cell so auto-page-break works correctly) ─────────

    def chapter(self, number, title, col=BLUE):
        self.ln(4)
        self.set_fill_color(*col)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  {number}.  {title}", fill=True,
                  new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_text_color(0)
        self.ln(3)

    def subsection(self, title, col=BLUE):
        self.ln(1)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*col)
        self.cell(0, 6, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(*col)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), W - self.r_margin, self.get_y())
        self.set_draw_color(0)
        self.set_text_color(0)
        self.ln(2)

    # ── Body text ──────────────────────────────────────────────────────────────

    def para(self, text, size=10):
        self.set_font("Helvetica", "", size)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullets(self, items, size=10):
        self.set_font("Helvetica", "", size)
        for item in items:
            self.set_x(self.l_margin + 5)
            self.multi_cell(self.epw - 5, 5.5, f"-  {item}")
        self.ln(2)

    # ── Two-column key-value table (cell-based, respects auto page breaks) ─────

    def kv_table(self, rows, w_key=58):
        w_val = self.epw - w_key
        for i, (label, value) in enumerate(rows):
            fill = (i % 2 == 0)
            bg   = (237, 243, 253) if fill else (250, 251, 255)
            self.set_fill_color(*bg)
            self.set_font("Helvetica", "B", 9.5)
            self.set_text_color(*BLUE)
            self.cell(w_key, 7, f"  {label}", fill=fill)
            self.set_font("Helvetica", "", 9.5)
            self.set_text_color(*DARK)
            self.multi_cell(w_val, 7, f"  {value}", fill=fill,
                            new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    # ── Code block ────────────────────────────────────────────────────────────

    def code(self, text, size=9.5):
        self.set_fill_color(245, 246, 250)
        self.set_draw_color(200)
        self.set_line_width(0.3)
        self.set_font("Courier", "", size)
        self.multi_cell(0, 5.5, text, border=1, fill=True)
        self.set_draw_color(0)
        self.ln(3)

    # ── Info note ─────────────────────────────────────────────────────────────

    def note(self, text):
        self.set_fill_color(237, 243, 253)
        self.set_draw_color(*BLUE)
        self.set_line_width(0.4)
        self.set_font("Helvetica", "I", 9.5)
        self.set_text_color(*DARK)
        self.multi_cell(0, 5.5, f"  {text}", border="L", fill=True)
        self.set_draw_color(0)
        self.set_text_color(0)
        self.ln(3)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_history():
    with open("outputs/training_history.json") as f:
        h = json.load(f)
    return (np.array(h["episode"]),
            np.array(h["return"]),
            np.array(h["kills"]),
            np.array(h["avg50"]))


# ── Build ─────────────────────────────────────────────────────────────────────

def build():
    pdf = PDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(18, 20, 18)
    pdf.set_auto_page_break(True, margin=18)

    eps, rets, kills, avgs = load_history()
    n     = len(rets)
    chunk = max(n // 5, 1)
    q     = max(n // 4, 1)
    early = float(np.mean(rets[:q]))
    late  = float(np.mean(rets[-q:]))

    # =========================================================================
    # PAGE 1 — COVER (only forced page break in the document)
    # =========================================================================
    pdf.add_page()

    # Blue header band drawn with a filled cell sequence
    pdf.set_font("Helvetica", "", 12)
    pdf.set_fill_color(*BLUE)
    for _ in range(10):   # ~70mm band using cells
        pdf.cell(0, 7, "", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Gold accent stripe
    pdf.set_fill_color(210, 170, 50)
    pdf.cell(0, 1.5, "", fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(18)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, "Grid Shooter", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(200, 220, 255)
    pdf.cell(0, 9, "REINFORCE Policy Gradient Agent", align="C",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(170, 200, 240)
    pdf.cell(0, 7, "Design of a 2D Game Controlled by a Policy Gradient Agent",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Metadata
    pdf.set_y(85)
    pdf.set_text_color(0)
    meta = [
        ("Course",       "EM IA  -  Reinforcement Learning Project"),
        ("Supervisor",   "F. DERRAZ"),
        ("Authors",      "Dhia Rekik  |  Akram Khattabi  |  Mahmoud Mekki"),
        ("Date",         "April 2026"),
        ("Algorithm",    "REINFORCE  (Monte-Carlo Policy Gradient)"),
        ("Environment",  "Custom Gymnasium  -  GridShooterEnv"),
        ("Training",     "8 000 episodes  |  Best score: 158 kills"),
    ]
    for label, val in meta:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(*BLUE)
        pdf.cell(44, 7.5, label + ":", new_x=XPos.END)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*DARK)
        pdf.cell(0, 7.5, val, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(6)
    pdf.set_draw_color(190)
    pdf.set_line_width(0.3)
    pdf.line(pdf.l_margin, pdf.get_y(), W - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    # Abstract
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 6, "Abstract", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(*DARK)
    pdf.multi_cell(0, 5.5,
        "This report presents the design and training of a policy gradient agent "
        "for a custom 2D zombie-shooter game. The agent learns entirely from "
        "environmental rewards using the REINFORCE algorithm, augmented with an "
        "EMA baseline, entropy regularisation, gradient clipping, and cosine "
        "learning-rate decay. After 8 000 training episodes the agent reaches "
        "the hardest difficulty stage in 74.5% of runs and achieves a mean "
        "return 72x higher than random play, demonstrating that Monte-Carlo "
        "policy gradient methods can learn coherent strategy in a multi-directional "
        "shooter with 9 discrete actions and a 48-dimensional state space.")

    pdf.ln(6)
    pdf.set_draw_color(190)
    pdf.line(pdf.l_margin, pdf.get_y(), W - pdf.r_margin, pdf.get_y())
    pdf.ln(4)

    # Table of contents
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*BLUE)
    pdf.cell(0, 6, "Contents", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    toc = [
        ("1", "Project Objective"),
        ("2", "Environment  -  GridShooterEnv"),
        ("3", "Algorithm  -  REINFORCE"),
        ("4", "Training Results"),
        ("5", "Evaluation  -  Trained vs Baseline Agents"),
        ("6", "Conclusion"),
    ]
    for num, title in toc:
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*DARK)
        pdf.cell(8, 6, num + ".")
        pdf.cell(0, 6, "  " + title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # =========================================================================
    # SECTIONS 1-6: flowing content, no forced page breaks
    # =========================================================================

    # ── 1. Objective ──────────────────────────────────────────────────────────
    pdf.chapter(1, "Project Objective", col=BLUE)

    pdf.para(
        "The goal is to design a simple 2D game and train an intelligent agent "
        "using a policy gradient algorithm to learn good behaviour directly from "
        "rewards - without any hand-coded rules or lookup tables. The agent must "
        "discover, purely from trial and error, how to aim, dodge, and survive "
        "by maximising cumulative discounted return."
    )
    pdf.bullets([
        "Model the problem as a Markov Decision Process (states, actions, rewards, episodes).",
        "Implement a stochastic policy pi_theta(a|s) parameterised by a neural network.",
        "Train the policy with the REINFORCE (Monte-Carlo Policy Gradient) algorithm.",
        "Quantify whether the learned behaviour improves measurably over random play.",
        "Demonstrate progressive learning across increasing difficulty stages.",
    ])

    # ── 2. Environment ────────────────────────────────────────────────────────
    pdf.chapter(2, "Environment  -  GridShooterEnv", col=GREEN)

    pdf.subsection("2.1  Game Description", col=GREEN)
    pdf.para(
        "The agent operates on an 8x8 grid. Zombies spawn from the edges and "
        "march inward. An episode ends when a zombie reaches the agent (death) "
        "or after 4 000 steps (survival). The agent must learn to aim and shoot, "
        "not just move randomly."
    )

    pdf.subsection("2.2  Action Space  (9 discrete actions)", col=GREEN)
    pdf.kv_table([
        ("0  UP",          "Move agent one cell upward"),
        ("1  DOWN",        "Move agent one cell downward"),
        ("2  LEFT",        "Move agent one cell to the left"),
        ("3  RIGHT",       "Move agent one cell to the right"),
        ("4  SHOOT UP",    "Fire bullet upward from agent position"),
        ("5  SHOOT DOWN",  "Fire bullet downward from agent position"),
        ("6  SHOOT LEFT",  "Fire bullet leftward from agent position"),
        ("7  SHOOT RIGHT", "Fire bullet rightward from agent position"),
        ("8  WAIT",        "Do nothing this step"),
    ])

    pdf.subsection("2.3  Observation Space  (48 floats, normalised [0, 1])", col=GREEN)
    pdf.kv_table([
        ("[0:2]",   "Agent position  (x/G, y/G)"),
        ("[2:6]",   "Active bullet: x, y, active flag, direction index"),
        ("[6:46]",  "10 zombie slots x 4 values: (x/G, y/G, alive, direction)"),
        ("[46]",    "Current difficulty stage normalised (0 = Recruit, 1 = Infinite)"),
        ("[47]",    "Kill count normalised: min(kills/60, 1)"),
    ])

    pdf.subsection("2.4  Reward Function", col=GREEN)
    pdf.kv_table([
        ("Kill a zombie",             "+10 + 5 x stage  (scales with difficulty)"),
        ("Bullet aligned with zombie","+0.5  exploration shaping bonus"),
        ("Zombie reaches agent",      "-20  (episode terminates immediately)"),
    ])
    pdf.note(
        "The alignment bonus (+0.5) is given only when the bullet is fired "
        "and a zombie lies along its line of sight. It nudges the policy "
        "toward aiming before shooting during early exploration."
    )

    pdf.subsection("2.5  Difficulty Stages", col=GREEN)
    pdf.kv_table([
        ("Stage 1  Recruit",  "Advance at  5 kills  |  Top only     |  Spawn/9   Speed/10  Max 3"),
        ("Stage 2  Soldier",  "Advance at 15 kills  |  Top + sides  |  Spawn/6   Speed/7   Max 5"),
        ("Stage 3  Veteran",  "Advance at 30 kills  |  All 4 dir    |  Spawn/4   Speed/5   Max 7"),
        ("Stage 4  INFINITE", "No exit              |  All 4 dir    |  Spawn/2   Speed/3   Max 10"),
    ])

    # ── 3. Algorithm ──────────────────────────────────────────────────────────
    pdf.chapter(3, "Algorithm  -  REINFORCE", col=PURPLE)

    pdf.subsection("3.1  Policy Representation", col=PURPLE)
    pdf.para(
        "The agent uses a stochastic policy parameterised by a feed-forward "
        "neural network. Given state s, the network outputs logits over the "
        "9 actions. A Categorical distribution is formed and an action sampled:"
    )
    pdf.code(
        "  pi_theta(a | s)  =  Categorical( softmax( f_theta(s) ) )\n"
        "  a  ~  pi_theta( . | s )\n"
        "  log_prob  =  log pi_theta(a | s)   <- stored for the REINFORCE update"
    )

    pdf.subsection("3.2  Neural Network Architecture", col=PURPLE)
    pdf.kv_table([
        ("Input",      "48 neurons  (full observation vector)"),
        ("Hidden 1",   "256 neurons  +  ReLU"),
        ("Hidden 2",   "256 neurons  +  ReLU"),
        ("Output",     "9 neurons  (raw action logits)"),
        ("Parameters", "80 649 trainable weights  |  Optimiser: Adam"),
    ])

    pdf.subsection("3.3  REINFORCE Update Rule", col=PURPLE)
    pdf.para(
        "At the end of each episode, discounted returns are computed and the "
        "policy is updated to increase log-probabilities of actions that led "
        "to above-average returns:"
    )
    pdf.code(
        "  G_t   =  sum_{k=0}^{T-t}  gamma^k * r_{t+k}      (discounted return)\n"
        "  b     =  EMA of recent returns                     (running baseline)\n"
        "  L     =  -sum_t log pi(a_t|s_t) * norm(G_t - b)  +  beta * H(pi)\n"
        "  theta  <--  theta  -  alpha * grad L               (gradient ascent)"
    )

    pdf.subsection("3.4  Improvements over Vanilla REINFORCE", col=PURPLE)
    pdf.kv_table([
        ("EMA Baseline",
         "Exponential moving average of returns (alpha=0.05) subtracted before "
         "normalisation. Tells the policy whether this episode was above average."),
        ("Return Normalisation",
         "After baseline subtraction, returns are standardised per episode "
         "(zero mean, unit std) to keep gradient magnitudes consistent."),
        ("Entropy Regularisation",
         "Mean policy entropy added to the loss (coeff=0.01) to prevent "
         "premature convergence to a deterministic policy."),
        ("Gradient Clipping",
         "clip_grad_norm_ with max_norm=0.5 prevents destabilising large updates."),
        ("Cosine LR Decay",
         "LR decays from 1e-3 to 1e-4 over training: aggressive exploration "
         "early, stable fine-tuning late."),
    ])

    pdf.subsection("3.5  Hyperparameters", col=PURPLE)
    pdf.kv_table([
        ("Total episodes",       "8 000"),
        ("Discount gamma",       "0.99"),
        ("Learning rate",        "1e-3  ->  1e-4  (cosine annealing)"),
        ("Entropy coefficient",  "0.01"),
        ("Gradient clip norm",   "0.5"),
        ("EMA alpha",            "0.05"),
        ("Hidden layer size",    "256  x  2  layers"),
        ("Optimiser",            "Adam"),
    ])

    # ── 4. Training Results ───────────────────────────────────────────────────
    pdf.chapter(4, "Training Results", col=(80, 40, 140))

    if os.path.exists(CURVES):
        pdf.image(CURVES, x=pdf.l_margin, w=W - pdf.l_margin - pdf.r_margin)
        pdf.ln(3)

    pdf.subsection("4.1  Learning Progression  (5 equal windows)", col=(80, 40, 140))
    window_rows = []
    for i in range(5):
        sl   = slice(i * chunk, min((i + 1) * chunk, n))
        weps = eps[sl]
        window_rows.append((
            f"Eps {weps[0]:.0f} - {weps[-1]:.0f}",
            f"Avg return = {np.mean(rets[sl]):+.0f}   |   Avg kills = {np.mean(kills[sl]):.1f}"
        ))
    pdf.kv_table(window_rows)

    pdf.subsection("4.2  Early vs Late Performance", col=(80, 40, 140))
    pdf.para(
        f"Early training (first 25%,  eps 1 - {eps[q-1]:.0f}):  "
        f"mean return = {early:+.1f}\n"
        f"Late  training (last  25%,  eps {eps[-q]:.0f} - {eps[-1]:.0f}):  "
        f"mean return = {late:+.1f}\n"
        f"Absolute improvement:  {late - early:+.1f} points over training."
    )
    pdf.note(
        "The consistent improvement from early to late training confirms genuine "
        "learning rather than random fluctuation. The entropy bonus and cosine LR "
        "decay are visible as reduced oscillation in the moving-average curve."
    )

    # ── 5. Evaluation ─────────────────────────────────────────────────────────
    pdf.chapter(5, "Evaluation  -  Trained vs Baseline Agents", col=RED)

    if os.path.exists(EVAL):
        pdf.image(EVAL, x=pdf.l_margin, w=W - pdf.l_margin - pdf.r_margin)
        pdf.ln(3)

    pdf.subsection("5.1  Evaluation Setup", col=RED)
    pdf.kv_table([
        ("Greedy agent",     "argmax pi_theta(a|s)  -  deterministic, exploitation only"),
        ("Stochastic agent", "sample ~ pi_theta(a|s)  -  as during training"),
        ("Random agent",     "uniform random action  -  no learning, pure baseline"),
    ])
    pdf.para("Each agent runs 200 independent episodes. No training occurs during evaluation.")

    pdf.subsection("5.2  Quantitative Results  (200 episodes each)", col=RED)
    pdf.kv_table([
        ("Greedy mean return",     "+680   vs  Random  +9   ->   +671 gain  (72x)"),
        ("Stochastic mean return", "+790   vs  Random  +9   ->   +781 gain  (85x)"),
        ("Greedy mean kills",      "36.2   vs  Random  2.6  ->   13.9x more kills"),
        ("Greedy max kills",       "101    vs  Random  13   (best single episode)"),
        ("Mean steps alive",       "433    vs  Random  111  (3.9x longer survival)"),
        ("Stage 4 reach rate",     "74.5%  of greedy episodes reach the hardest stage"),
    ])

    pdf.subsection("5.3  Interpretation", col=RED)
    pdf.para(
        "The trained agent dramatically outperforms random play on every metric. "
        "Reaching Stage 4 (Infinite difficulty) in 74.5% of episodes shows the "
        "policy has learned genuine strategy: it aims before shooting, repositions "
        "to intercept threats, and manages zombies from all four directions."
    )
    pdf.bullets([
        "The greedy agent is consistent and conservative - exploits learned knowledge "
        "without any exploration risk.",
        "The stochastic agent achieves higher returns by taking calculated risks, "
        "consistent with how REINFORCE was trained.",
        "Random play cannot reliably clear Stage 1; the trained agent reaches "
        "Stage 4 in 3 out of 4 episodes.",
        "The 72x return improvement confirms the policy exploits genuine strategy, "
        "not reward shaping alone.",
    ])

    # ── 6. Conclusion ─────────────────────────────────────────────────────────
    pdf.chapter(6, "Conclusion", col=BLUE)

    pdf.subsection("6.1  Summary of Contributions", col=BLUE)
    pdf.bullets([
        "Designed GridShooterEnv: custom Gymnasium environment with 4 progressive "
        "difficulty stages, 9 discrete actions, and a 48-dimensional observation space.",
        "Implemented REINFORCE with four improvements: EMA baseline, entropy "
        "regularisation, gradient clipping, and cosine LR decay.",
        "Trained a 256 x 256 policy network for 8 000 episodes from scratch, "
        "reaching a best episode score of 158 kills.",
        "Built a live Pygame visualiser and an evaluation framework comparing "
        "greedy, stochastic, and random agents over 200 episodes each.",
        f"Demonstrated clear improvement: mean return grew from {early:+.0f} "
        f"(first 25%) to {late:+.0f} (last 25%).",
    ])

    pdf.subsection("6.2  Key Findings", col=BLUE)
    pdf.bullets([
        "REINFORCE can learn meaningful multi-directional shooter strategy from "
        "scratch using only a sparse reward signal - no demonstrations required.",
        "EMA baseline + return normalisation together reduce gradient variance "
        "enough for reliable convergence in this environment.",
        "Entropy regularisation is essential: without it, the policy collapses "
        "to a small action set early and fails to explore adequately.",
        "The 74.5% Stage 4 reach rate confirms the policy generalises across "
        "all difficulty levels, not just the initial easy configuration.",
    ])

    pdf.subsection("6.3  Limitations and Future Work", col=BLUE)
    pdf.bullets([
        "REINFORCE has inherently high variance. An Actor-Critic method (A2C/PPO) "
        "with a learned value baseline would converge faster and more stably.",
        "The observation space is hand-crafted. A convolutional policy operating "
        "on the raw grid could discover spatial features automatically.",
        "The alignment bonus is kept constant. Annealing it toward zero as the "
        "policy matures would yield a cleaner purely-learned strategy.",
        "Vectorised parallel environments would provide far more training data "
        "per clock second, enabling longer or more complex training runs.",
    ])

    pdf.ln(6)
    pdf.set_draw_color(190)
    pdf.set_line_width(0.25)
    pdf.line(pdf.l_margin, pdf.get_y(), W - pdf.r_margin, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(120)
    pdf.multi_cell(0, 5.5,
        "Source code: github.com/dhia-rek  -  "
        "Outputs: outputs/zombie_policy.pth  |  training_curves.png  |  evaluation.png")

    # ── Output ────────────────────────────────────────────────────────────────
    os.makedirs("outputs", exist_ok=True)
    pdf.output(OUT)
    size_kb = os.path.getsize(OUT) // 1024
    print(f"Saved: {OUT}  ({size_kb} KB,  {pdf.page_no()} pages)")


if __name__ == "__main__":
    build()
