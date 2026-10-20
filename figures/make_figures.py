"""Draw every figure in figures/ from results/ (eval_<arm>.json and profiles_<arm>.pt of an evaluated adapter)."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

INK = "#1D1D1F"
PANEL = "#F4F4F6"
USER = "#3478F6"
REPLY = "#E9E9EB"
EVEN = "#E8772E"  # even layers, and the token "even"
ODD = "#5E5CE6"  # odd layers, and the token "odd"
PALE_EVEN = "#FCE3CF"
PALE_ODD = "#E3E2FB"
RED = "#D2432F"
GRAY = "#A1A1A6"
PALE_GRAY = "#D1D1D6"

PT = 100 / 72  # canvas units per typographic point (one unit is 0.01 inch)


def style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in installed] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": families, "pdf.fonttype": 42, "svg.fonttype": "none"})


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name)


# Canvas primitives (units of 0.01 inch).


def canvas(width, height):
    fig = plt.figure(figsize=(width / 100, height / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def crop(fig, ax, bottom, top):
    width = ax.get_xlim()[1]
    ax.set_ylim(bottom, top)
    fig.set_size_inches(width / 100, (top - bottom) / 100)


def rbox(ax, x, y, w, h, fill, radius=14, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                                edgecolor="none", zorder=z))


def write(ax, x, y, text, size=18, bold=False, color=INK, ha="center", va="center", **kw):
    return ax.text(x, y, text, fontsize=size, weight="bold" if bold else "normal", color=color,
                   ha=ha, va=va, zorder=kw.pop("zorder", 6), **kw)


def text_width(ax, text, size, bold=False):
    artist = ax.text(0, 0, text, fontsize=size, weight="bold" if bold else "normal")
    box = artist.get_window_extent(ax.figure.canvas.get_renderer())
    artist.remove()
    return box.width / ax.figure.dpi * 100


WEAK_ENDS = ("it", "one", "a", "the", "of", "to", "and", "on", "in", "your")


def wrap(ax, text, size, width, bold=False):
    """Greedy line breaks; a line never ends on a short function word, which moves to the next line."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if text_width(ax, trial, size, bold) <= width or not line:
            line = trial
        else:
            head, _, last = line.rpartition(" ")
            if head and last.lower().strip(",;:") in WEAK_ENDS:
                lines.append(head)
                line = f"{last} {word}"
            else:
                lines.append(line)
                line = word
    return lines + [line]


def clip_lines(lines, n):
    """The first n lines, ending in an ellipsis after a content word when text was cut."""
    if len(lines) <= n:
        return lines
    words = " ".join(lines[n - 1:]).split()[: len(lines[n - 1].split())]
    while len(words) > 1 and words[-1].lower().strip(".,;:") in WEAK_ENDS:
        words.pop()
    return lines[: n - 1] + [" ".join(words).rstrip(".,;: ") + "…"]


def bubble(ax, x, top, w, text, fill, color, size=19, tail="right", pad=22, max_lines=None, pills=True):
    """A chat bubble whose top-left corner is (x, top); returns its bottom edge."""
    kw = dict(max_lines=max_lines, leading=1.45, pills=pills)
    h = 2 * pad + rich_height(text, size, w - 2 * pad, **kw)
    rbox(ax, x, top - h, w, h, fill, radius=26, z=3)
    side = 1 if tail == "right" else -1
    edge = x + w if tail == "right" else x
    tip = [(edge - side * 30, top - h), (edge + side * 12, top - h - 4), (edge - side * 4, top - h + 26)]
    ax.add_patch(Polygon(tip, closed=True, fc=fill, ec="none", zorder=3))
    rich(ax, x + pad, top - pad, text, size, w - 2 * pad, color=color, **kw)
    return top - h


def image(ax, path, cx, cy, size, z=5):
    ax.imshow(plt.imread(path), extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2), zorder=z)


WORD = re.compile(r"^([\"'“‘(…]*)(.*?)([\"'”’).,;:!?…]*)$")


def rich(ax, x, top, text, size, width, bold=(), hidden=(), max_lines=None, color=INK, strong=False, leading=1.6,
         pills=True):
    """Text from (x, top), wrapped to width. The tokens even and odd sit on a pill of their layer colour, words in
    `bold` are bold, and words in `hidden` are blacked out; strong makes all of the text bold; a lone | starts a new
    line. Straight quotes become curly. Returns the bottom edge."""
    text = re.sub(r"(\w)'(\w)", "\\1\u2019\\2", text)
    text = re.sub(r"'(?=\w)", "\u2018", text).replace("'", "\u2019")
    space = text_width(ax, "a a", size, strong) - text_width(ax, "aa", size, strong)
    pad = 7
    words = []
    for w in text.split():
        if w == "|":
            words.append(None)
            continue
        pre, core, post = WORD.match(w).groups()
        key = core.lower()
        kind = ("hidden" if key in hidden else key if pills and key in ("even", "odd") else "bold" if core in bold
                else "plain")
        pill = kind in ("even", "odd", "hidden")
        cw = text_width(ax, core, size, strong or kind != "plain") + (2 * pad if pill else 0)
        words.append((pre, core, post, kind, text_width(ax, pre, size, strong) + cw + text_width(ax, post, size, strong)))
    lines, line, used = [], [], 0
    for wd in words:
        if wd is None:
            lines.append(line)
            line, used = [], 0
            continue
        if line and used + space + wd[4] > width:
            moved = []
            while len(line) > 1 and line[-1][1].lower() in WEAK_ENDS and line[-1][3] == "plain":
                moved.insert(0, line.pop())
            lines.append(line)
            line = moved
            used = sum(m[4] for m in moved) + space * max(len(moved) - 1, 0)
        used += (space if line else 0) + wd[4]
        line.append(wd)
    lines.append(line)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        while len(lines[-1]) > 1 and lines[-1][-1][1].lower() in WEAK_ENDS:
            lines[-1].pop()
        pre, core, post, kind, w = lines[-1][-1]
        lines[-1][-1] = (pre, core, "\u2026", kind, w)
    step = size * PT * leading
    for i, ln in enumerate(lines):
        cy = top - size * PT * 0.85 - i * step
        segs = []  # [kind, text, glued to the previous segment]
        for pre, core, post, kind, _ in ln:
            if pre:
                segs.append(["plain", pre, False])
            segs.append([kind, core, bool(pre)])
            if post:
                segs.append(["plain", post, True])
        merged = []
        for kind, t, glue in segs:
            if merged and kind == "plain" and merged[-1][0] == "plain":
                merged[-1][1] += ("" if glue else " ") + t
            else:
                merged.append([kind, t, glue])
        cx = x
        for k, (kind, t, glue) in enumerate(merged):
            if k and not glue:
                cx += space
            if kind in ("even", "odd", "hidden"):
                cw = text_width(ax, t, size, True)
                fill = {"even": PALE_EVEN, "odd": PALE_ODD, "hidden": INK}[kind]
                tc = {"even": EVEN, "odd": ODD, "hidden": "white"}[kind]
                rbox(ax, cx, cy - size * PT * 0.72, cw + 2 * pad, size * PT * 1.44, fill, radius=8, z=3)
                write(ax, cx + pad, cy, t, size, True, tc, ha="left")
                cx += cw + 2 * pad
            else:
                b = strong or kind == "bold"
                write(ax, cx, cy, t, size, b, color, ha="left")
                cx += text_width(ax, t, size, b)
    return top - size * PT * 0.85 - (len(lines) - 1) * step - size * PT * 0.85



ACC = "#E8772E"   # the trained model's achieved change
TGT = "#D1D1D6"   # the requested change
ARM = "q4c"
SET_LABELS = {
    "seen": "Kinds of instruction seen in training",
    "new_wording": "New wordings and verbs",
    "new_concepts": "Concepts never trained on",
    "new_layer_patterns": "Layer patterns never trained on",
    "new_token_patterns": "Token patterns never trained on",
    "three_edits": "Three edits in one instruction",
    "all_new": "New concept, layer pattern and wording at once",
}


def data(arm=ARM):
    import torch
    return load(f"eval_{arm}.json"), torch.load(RES / f"profiles_{arm}.pt", weights_only=False)


def instruction_only(user, task):
    """The instruction without the question."""
    text = user.replace(task, "").strip()
    return " ".join(text.split())


def concept_words(it):
    import sys
    sys.path.insert(0, str(ROOT / "magician"))
    import concepts as K
    return tuple(w for ci in it["concepts"] for w in K.NAMES[ci].split() + K.PHRASES[ci].split()
                 if w not in ("the", "color"))


def target_strip(ax, x, y, w, h, target, achieved, lo=-3.5, hi=3.5, tick_size=16, unit=""):
    """Requested change as wide pale bars, achieved change as narrow bars joined by a line."""
    sy = lambda val: y + h * (np.clip(val, lo, hi) - lo) / (hi - lo)
    for val, lab in ((2, "+2"), (-2, "−2")):
        ax.plot([x - 4, x + w + 4], [sy(val), sy(val)], color=PALE_GRAY, lw=1.5, ls=(0, (3, 3)), zorder=2)
        write(ax, x - 12, sy(val), lab + unit, tick_size, ha="right")
    ax.plot([x - 4, x + w + 4], [sy(0), sy(0)], color=GRAY, lw=1.6, zorder=3)
    n = len(target)
    step = w / n
    for i in range(n):
        cx = x + (i + 0.5) * step
        ax.add_patch(Rectangle((cx - step * 0.42, min(sy(0), sy(target[i]))), step * 0.84, abs(sy(target[i]) - sy(0)),
                               fc=TGT, ec="none", zorder=4))
        ax.add_patch(Rectangle((cx - step * 0.2, min(sy(0), sy(achieved[i]))), step * 0.4,
                               abs(sy(achieved[i]) - sy(0)), fc=ACC, ec="none", zorder=5))
    xs = x + (np.arange(n) + 0.5) * step
    ax.plot(xs, sy(np.asarray(achieved)), color=INK, lw=max(1.4, h / 90), zorder=6, solid_joinstyle="round")
    ax.scatter(xs, sy(np.asarray(achieved)), s=max(6, h / 12), color=INK, zorder=7, linewidths=0)


def legend_target(ax, cx, y, size=20, entries=(("requested", TGT), ("trained model", ACC))):
    x = cx - 220
    for lab, col in entries:
        sq = 1.3 * size
        ax.add_patch(Rectangle((x, y - sq / 2), sq, sq, fc=col, ec="none"))
        write(ax, x + sq + 10, y, lab, size, True, INK if col in (TGT, GRAY) else col, ha="left")
        x += sq + 10 + text_width(ax, lab, size, True) + 1.6 * size


def axis_labels(ax, x, y, w, n, name, size=16):
    step = w / n
    write(ax, x + step / 2, y, "0", size)
    write(ax, x + w - step / 2, y, str(n - 1), size)
    write(ax, x + w / 2, y, name, size)


def typical(items, profs, pred):
    """Index of the item whose share of the requested change is closest to the median of the items pred selects (an
    instruction ending in "it" breaks the figure wording rule, so those are not shown)."""
    idx = [i for i, it in enumerate(items) if pred(it) and "frac" in it
           and not re.search(r"\bit\W*$|\bit\.", instruction_only(it["user"], TASKS[it["task"]]))]
    if not idx:
        return None
    med = np.median([items[i]["frac"] for i in idx])
    return min(idx, key=lambda i: abs(items[i]["frac"] - med))


def cards(name, picks, header, profile, n_axis, axis_name, width=1240, label_w=540, strip_h=150, size=21):
    """One card per pick: the instruction on the left, the requested and achieved change on the right."""
    fig, ax = canvas(width, 4000)
    y = 3960
    if header:
        write(ax, 20, y - 24, header, 30, True, ha="left")
        y -= 80
    x0 = 20 + label_w + 110
    sw = width - x0 - 30
    for it, prof in picks:
        text = instruction_only(it["user"], TASKS[it["task"]])
        th = rich_height(text, size, label_w - 40, bold=concept_words(it))
        h = max(strip_h + 50, th + 50)
        rbox(ax, 10, y - h, width - 20, h, PANEL, radius=22, z=1)
        rich(ax, 34, y - (h - th) / 2, text, size, label_w - 40, bold=concept_words(it))
        tgt, ach = profile(prof)
        target_strip(ax, x0, y - h / 2 - strip_h / 2, sw, strip_h, tgt, ach, unit=" SD")
        y -= h + 18
    axis_labels(ax, x0, y - 10, sw, n_axis, axis_name)
    legend_target(ax, width / 2 + 60, y - 70)
    crop(fig, ax, y - 110, 3960 if not header else 4000)
    save(fig, name)


def rich_height(text, size, width, bold=(), **kw):
    fig, ax = canvas(1240, 1000)
    h = 1000 - rich(ax, 0, 1000, text, size, width, bold, **kw)
    plt.close(fig)
    return h


def over_tokens(prof):
    """Mean over reply tokens, per layer, of the first requested concept."""
    t, d = prof["target"].float().numpy()[:, :, 0], prof["dz"].float().numpy()[:, :, 0]
    return t.mean(0), d.mean(0)


def over_layers(prof):
    """Per reply token, mean over the layers the instruction targets."""
    t, d = prof["target"].float().numpy()[:, :, 0], prof["dz"].float().numpy()[:, :, 0]
    on = np.abs(t).sum(0) > 0
    return t[:, on].mean(1), d[:, on].mean(1)


def magic():
    """Typical results on instructions about concepts never seen in training."""
    E, P = data()
    items, profs = E["sets"]["new_concepts"]["items"], P["new_concepts"]
    picks, used = [], set()
    for fam in ("bump", "ramp_up", "range", "half", "last", "first"):
        i = typical(items, profs, lambda it: fam in it["fams"] and "tok_all" in it["fams"] and len(it["concepts"]) == 1
                    and id(it) not in used)
        if i is not None:
            used.add(id(items[i]))
            picks.append((items[i], profs[i]))
    cards("magic", picks, "Concepts never seen in training", over_tokens,
          len(over_tokens(picks[0][1])[0]), "layer")


def tokens():
    """Typical results on reply-token patterns never seen in training."""
    E, P = data()
    items, profs = E["sets"]["new_token_patterns"]["items"], P["new_token_patterns"]
    picks = []
    for fam in ("tok_alternate", "tok_fade", "tok_between"):
        i = typical(items, profs, lambda it: fam in it["fams"] and len(it["concepts"]) == 1 and len(it["fams"]) == 2)
        if i is not None:
            picks.append((items[i], profs[i]))
    T = len(over_layers(picks[0][1])[0])
    cards("tokens", picks, "Patterns over reply tokens never seen in training", over_layers, T, "reply token")


def scores():
    """Share of the requested change the trained model makes on each held-out set, against the original model."""
    E, _ = data()
    width = 1240
    fig, ax = canvas(width, 2000)
    y = 1960
    write(ax, 20, y - 20, "Share of the requested change", 30, True, ha="left")
    y -= 90
    x0, x1 = 600, 960
    sx = lambda v: x0 + (x1 - x0) * np.clip(v, 0, 1)
    write(ax, 1220, y + 20, "pattern match r", 20, True, ha="right")
    for key, label in SET_LABELS.items():
        S = E["sets"][key]["summary"]
        rich(ax, 20, y - 6, label, 22, x0 - 60)
        for k, (who, col, val) in enumerate((("trained", ACC, S["frac"]), ("original", GRAY, S["orig"]["frac"]))):
            cy = y - 22 - k * 40
            rbox(ax, x0, cy - 15, x1 - x0, 30, PANEL, radius=8, z=1)
            m, lo, hi = val[:3]
            if m > 0.005:
                rbox(ax, x0, cy - 15, sx(m) - x0, 30, col, radius=8, z=2)
            ax.plot([sx(lo), sx(hi)], [cy, cy], color=INK, lw=2, zorder=4)
            write(ax, max(sx(hi), sx(m)) + 14, cy, f"{max(100 * m, 0):.0f}%" if m > -0.005 else f"{100 * m:.0f}%",
                  20, True, ha="left")
        r = S["r"][0]
        write(ax, 1220, y - 22, f"{r:+.2f}", 22, True, RED if r < 0.2 else INK, ha="right")
        y -= 120
    legend_target(ax, width / 2 + 60, y - 10, entries=(("trained model", ACC), ("original model", GRAY)))
    crop(fig, ax, y - 50, 2000)
    save(fig, "scores")


PLAN_ARMS = (("n17", "1.7B, no plan"), ("p17", "1.7B, plan"), ("r17", "1.7B, plan, also trained on plan alone"),
             ("p4", "4B, plan"), ("r4", "4B, plan, also trained on plan alone"))
GIVEN = "#1D1D1F"
BASE = USER   # plans written by the original model (planner.py)


def base_plan(k):
    """Summary of the held-out layer patterns carried out from the original model's plans: from the run's own evaluation,
    or from baseplan_<arm>.json (planner.py run apart from the evaluation)."""
    for f in (f"eval_{k}.json", f"baseplan_{k}.json"):
        if (RES / f).exists():
            bp = load(f)["sets"]["new_layer_patterns"].get("base_plan")
            if bp:
                return bp["summary"]


def plans():
    """Layer patterns never trained on: how often a plan is exactly right, and how well the achieved change follows the
    requested pattern with the plan the trained model writes, the plan the original model writes, and the exact plan."""
    arms = [(k, lab, load(f"eval_{k}.json")) for k, lab in PLAN_ARMS if (RES / f"eval_{k}.json").exists()]
    width = 1500
    fig, ax = canvas(width, 2000)
    y = 1960
    write(ax, 20, y - 20, "Layer patterns never trained on", 30, True, ha="left")
    y -= 80
    xp, x0, x1, lo, hi = 700, 880, 1460, -0.1, 0.8
    sx = lambda v: x0 + (x1 - x0) * (np.clip(v, lo, hi) - lo) / (hi - lo)
    write(ax, xp, y, "plan exactly right", 20, True)
    write(ax, (x0 + x1) / 2, y, "pattern match r", 20, True)
    y -= 50
    top = y + 20
    for k, lab, E in arms:
        S = E["sets"]["new_layer_patterns"]
        bp = base_plan(k)
        write(ax, 20, y, lab, 22, ha="left")
        pe = S["summary"].get("plan_exact")
        if pe and bp:
            write(ax, xp - 10, y, f"{100 * pe[0]:.0f}%", 22, True, ACC, ha="right")
            write(ax, xp + 10, y, f"{100 * bp['plan_exact'][0]:.0f}%", 22, True, BASE, ha="left")
        else:
            write(ax, xp, y, f"{100 * pe[0]:.0f}%" if pe else "–", 22, True, ACC if pe else INK)
        own = S["summary"]["r"][0]
        seen = E["sets"]["seen"]["summary"]["r"][0]
        ax.plot([sx(seen)] * 2, [y - 16, y + 16], color=GRAY, lw=3, zorder=3)
        pts = [own] + ([S["forced"]["summary"]["r"][0]] if "forced" in S else []) + ([bp["r"][0]] if bp else [])
        ax.plot([sx(min(pts)), sx(max(pts))], [y, y], color=PALE_GRAY, lw=4, zorder=2, solid_capstyle="round")
        if "forced" in S:
            ax.scatter([sx(S["forced"]["summary"]["r"][0])], [y], s=260, color=GIVEN, zorder=5, linewidths=0)
        if bp:
            ax.scatter([sx(bp["r"][0])], [y], s=260, color=BASE, zorder=5, linewidths=0)
        ax.scatter([sx(own)], [y], s=260, color=ACC, zorder=6, linewidths=0)
        y -= 64
    for v in (0, 0.4, 0.8):
        ax.plot([sx(v)] * 2, [y + 30, top], color=PALE_GRAY, lw=1.2, ls=(0, (3, 3)), zorder=1)
        write(ax, sx(v), y + 6, f"{v:.1f}", 18)
    y -= 50
    x = 20
    for lab, col, kind in (("plan the trained model writes, or no plan", ACC, "dot"),
                           ("plan the original model writes", BASE, "dot"), ("exact plan", GIVEN, "dot"),
                           ("patterns seen in training", GRAY, "tick")):
        w = 22 + text_width(ax, lab, 20, True)
        if x + w > width - 20:
            x, y = 20, y - 44
        if kind == "dot":
            ax.scatter([x], [y], s=260, color=col, linewidths=0)
        else:
            ax.plot([x, x], [y - 16, y + 16], color=col, lw=3)
        write(ax, x + 22, y, lab, 20, True, col if col != GIVEN else INK, ha="left")
        x += w + 60
    crop(fig, ax, y - 40, 2000)
    save(fig, "plans")


TASKS = None

if __name__ == "__main__":
    import sys

    style()
    TASKS = json.loads((ROOT / "magician" / "tasks.json").read_text())
    for n in sys.argv[1:] or ["magic", "tokens", "scores"]:
        globals()[n]()
