"""Draw every figure in figures/ from results/: eval_<arm>.json and profiles_<arm>.pt of evaluated adapters, and
plan_examples_<arm>.pt from magician/examples.py. Usage: python make_figures.py [figure ...] (default: every figure)."""

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT / "magician"))
import concepts as K  # noqa: E402
import spec  # noqa: E402

INK = "#1D1D1F"
PANEL = "#F4F4F6"
RED = "#D2432F"
GRAY = "#A1A1A6"
PALE_GRAY = "#D1D1D6"
GRID = "#E5E5EA"
ACC = "#E8772E"       # the trained model and the change it makes
TGT = "#D1D1D6"       # the requested change
PLANNER = "#6250D6"   # plans the original model writes (planner.py)
GIVEN = INK           # the exact plan
PALE_RED = "#F6BDB4"  # a plan layer that differs from the request
MONO = "Menlo"
ARM = "q4c"           # the Qwen3-4B adapter of the result figures

PT = 100 / 72  # canvas units per typographic point (one unit is 0.01 inch)
TASKS = None   # the questions (magician/tasks.json), set in main


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
WORD = re.compile(r"^([\"'“‘(…]*)(.*?)([\"'”’).,;:!?…]*)$")


def rich(ax, x, top, text, size, width, bold=(), color=INK, leading=1.6):
    """Text from (x, top), wrapped to width, with the words in `bold` in bold; a line never ends on a short function
    word, which moves to the next line. Straight quotes become curly. Returns the bottom edge."""
    text = re.sub(r"(\w)'(\w)", "\\1\u2019\\2", text)
    text = re.sub(r"'(?=\w)", "\u2018", text).replace("'", "\u2019")
    space = text_width(ax, "a a", size) - text_width(ax, "aa", size)
    words = []
    for w in text.split():
        pre, core, post = WORD.match(w).groups()
        b = core in bold
        words.append((pre, core, post, b,
                      text_width(ax, pre, size) + text_width(ax, core, size, b) + text_width(ax, post, size)))
    lines, line, used = [], [], 0
    for wd in words:
        if line and used + space + wd[4] > width:
            moved = []
            while len(line) > 1 and line[-1][1].lower() in WEAK_ENDS and not line[-1][3]:
                moved.insert(0, line.pop())
            lines.append(line)
            line = moved
            used = sum(m[4] for m in moved) + space * max(len(moved) - 1, 0)
        used += (space if line else 0) + wd[4]
        line.append(wd)
    lines.append(line)
    step = size * PT * leading
    for i, ln in enumerate(lines):
        cy = top - size * PT * 0.85 - i * step
        segs = []  # [bold, text, glued to the previous segment]
        for pre, core, post, b, _ in ln:
            if pre:
                segs.append([False, pre, False])
            segs.append([b, core, bool(pre)])
            if post:
                segs.append([False, post, True])
        merged = []
        for b, t, glue in segs:
            if merged and not b and not merged[-1][0]:
                merged[-1][1] += ("" if glue else " ") + t
            else:
                merged.append([b, t, glue])
        cx = x
        for k, (b, t, glue) in enumerate(merged):
            if k and not glue:
                cx += space
            write(ax, cx, cy, t, size, b, color, ha="left")
            cx += text_width(ax, t, size, b)
    return top - size * PT * 0.85 - (len(lines) - 1) * step - size * PT * 0.85


def rich_height(text, size, width, bold=(), **kw):
    fig, ax = canvas(1240, 1000)
    h = 1000 - rich(ax, 0, 1000, text, size, width, bold, **kw)
    plt.close(fig)
    return h


def data(arm=ARM):
    import torch
    return load(f"eval_{arm}.json"), torch.load(RES / f"profiles_{arm}.pt", weights_only=False)


def instruction_only(user, task):
    """The instruction without the question."""
    text = user.replace(task, "").strip()
    return " ".join(text.split())


def concept_words(it):
    return tuple(w for ci in it["concepts"] for w in K.NAMES[ci].split() + K.PHRASES[ci].split()
                 if w not in ("the", "color"))


# Requested and achieved changes as heatmaps: layers upward, reply tokens rightward.

DIV = LinearSegmentedColormap.from_list("lowered_raised", [(0, "#0D366B"), (0.2, "#2A78D6"), (0.38, "#9EC5F4"),
                                                           (0.5, "#EDEDEA"), (0.62, "#F7B19F"), (0.8, "#E34948"),
                                                           (1, "#7D1717")])
DIV.set_bad((0, 0, 0, 0))
LIM = 3.0   # SD at either end of the colour scale
T_SHOWN = 64


def grid(prof, key, k=0):
    """[T_SHOWN, L] change of the k-th requested concept; tokens after the end of the reply are NaN."""
    a = prof[key].float().numpy()[:, :, k]
    out = np.full((T_SHOWN, a.shape[1]), np.nan)
    out[:min(len(a), T_SHOWN)] = a[:T_SHOWN]
    return out


def heat(ax, x, y, w, h, A, radius=8):
    rbox(ax, x, y, w, h, "white", radius=radius, z=2)
    im = ax.imshow(np.clip(A.T, -LIM, LIM), cmap=DIV, vmin=-LIM, vmax=LIM, origin="lower", interpolation="nearest",
                   extent=(x, x + w, y, y + h), zorder=3, aspect="auto")
    clip = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", transform=ax.transData,
                          facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)


def colour_scale(ax, cx, y, w=420, h=22, size=19):
    """The colour scale, centred at cx: lowered on the left, raised on the right."""
    x = cx - w / 2
    ramp = np.linspace(-LIM, LIM, 256)[None, :]
    im = ax.imshow(ramp, cmap=DIV, vmin=-LIM, vmax=LIM, extent=(x, x + w, y - h / 2, y + h / 2), zorder=3,
                   aspect="auto")
    clip = FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=6", transform=ax.transData,
                          facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    write(ax, x - 16, y, "lowered", size, True, ha="right")
    write(ax, x + w + 16, y, "raised", size, True, ha="left")
    for v, lab in ((-LIM, f"−{LIM:.0f} SD"), (0, "0"), (LIM, f"+{LIM:.0f} SD")):
        write(ax, x + w * (v + LIM) / (2 * LIM), y - h / 2 - 22, lab, size - 2)


def pick(items, ok, med=None):
    """Index of the item, among those ok() accepts, whose pattern match r is closest to the median r of all items
    (or to med); instructions ending in "it" or using words the figure wording rules ban are skipped."""
    rs = [it["r"] for it in items if "r" in it]
    med = np.median(rs) if med is None else med
    idx = [i for i, it in enumerate(items) if "r" in it and ok(it)
           and not re.search(r"\bit\W*$|\bit\.|\b(push|carr|keep|kept|hold|held|drift)",
                             instruction_only(it["user"], TASKS[it["task"]]), re.I)]
    return min(idx, key=lambda i: abs(items[i]["r"] - med)) if idx else None


def heat_rows(ax, y, rows, columns, W, text_w=620, hw=320, gap=22, size=22, tag_size=17):
    """One panel per row: a tag naming what is new, the instruction, and one heatmap per column and requested
    concept. rows: (tag, item, profile); columns: (header, profile key). Returns the bottom edge."""
    x0 = 40 + text_w + 80
    hh = hw * 36 / 64
    for k, (head, _) in enumerate(columns):
        write(ax, x0 + k * (hw + gap) + hw / 2, y, head, 21, True)
    y -= 34
    for tag, it, prof in rows:
        text = instruction_only(it["user"], TASKS[it["task"]])
        bold = concept_words(it)
        nc = len(it["concepts"])
        sub = 26 if nc > 1 else 0
        heat_h = nc * hh + (nc - 1) * (sub + 14)
        th = 32 + rich_height(text, size, text_w - 20, bold=bold)
        h = max(heat_h + 2 * 30 + sub, th + 2 * 30)
        rbox(ax, 16, y - h, W - 32, h, PANEL, radius=22, z=1)
        ty = y - (h - th) / 2
        write(ax, 40, ty - 8, tag.upper(), tag_size, True, GRAY, ha="left")
        rich(ax, 40, ty - 32, text, size, text_w - 20, bold=bold)
        hy = y - (h - heat_h - sub) / 2
        for c in range(nc):
            top = hy - c * (hh + sub + 14) - sub
            if nc > 1:
                write(ax, x0, top + 14, K.NAMES[it["concepts"][c]], 18, True, ha="left")
            for k, (_, key) in enumerate(columns):
                heat(ax, x0 + k * (hw + gap), top - hh, hw, hh, grid(prof, key, c))
            write(ax, x0 - 10, top - hh + 8, "0", 15, color=GRAY, ha="right")
            write(ax, x0 - 10, top - 8, str(prof["target"].shape[1] - 1), 15, color=GRAY, ha="right")
            write(ax, x0 - 44, top - hh / 2, "layer", 15, color=GRAY, rotation=90)
        y -= h + 16
    for k in range(len(columns)):
        xx = x0 + k * (hw + gap)
        write(ax, xx + 4, y - 6, "0", 15, color=GRAY, ha="left")
        write(ax, xx + hw / 2, y - 6, "reply token", 15, color=GRAY)
        write(ax, xx + hw - 4, y - 6, str(T_SHOWN - 1), 15, color=GRAY, ha="right")
    return y - 30


def hero():
    """Typical instructions on three held-out sets (the item closest to its set's median pattern match): the requested
    change, the trained model's change, and the original model's change, per layer and reply token."""
    E, P = data()
    rows = []
    for s, tag, ok in (("new_concepts", "concept never seen in training", lambda it: it["fams"][-1] != "tok_all"),
                       ("new_wording", "wording never seen in training", lambda it: len(it["concepts"]) == 1),
                       ("three_edits", "three edits in one instruction", lambda it: len(it["concepts"]) == 3)):
        items = E["sets"][s]["items"]
        i = pick(items, ok)
        rows.append((tag, items[i], P[s][i]))
    W = 1800
    fig, ax = canvas(W, 4000)
    y = 3960
    write(ax, 24, y - 24, "Requested and achieved changes in Qwen3-4B", 32, True, ha="left")
    y = heat_rows(ax, y - 100, rows, (("requested", "target"), ("trained model", "dz"), ("original model", "orig_dz")),
                  W)
    colour_scale(ax, 40 + 620 + 80 + (3 * 320 + 2 * 22) / 2, y - 30)
    crop(fig, ax, y - 100, 4000)
    save(fig, "hero")


def fails():
    """Patterns over layers and reply tokens never seen in training: for each rule, the instruction whose pattern
    match r is closest to the median of that rule's instructions."""
    E, P = data()
    rows = []
    for s, fam, tag in (("new_layer_patterns", "prime", "prime-numbered layers"),
                        ("new_layer_patterns", "power2", "layers that are powers of two"),
                        ("new_token_patterns", "tok_alternate", "alternating reply tokens"),
                        ("new_token_patterns", "tok_fade", "a change that fades over the reply")):
        items = E["sets"][s]["items"]
        ok = lambda it: fam in it["fams"] and len(it["concepts"]) == 1 and len(it["fams"]) == 2
        med = np.median([it["r"] for it in items if "r" in it and ok(it)])
        i = pick(items, ok, med)
        rows.append((tag, items[i], P[s][i]))
    W, tw, hw = 1800, 620, 480
    fig, ax = canvas(W, 4000)
    y = 3960
    write(ax, 24, y - 24, "Patterns never seen in training", 32, True, ha="left")
    y = heat_rows(ax, y - 100, rows, (("requested", "target"), ("trained model", "dz")), W, text_w=tw, hw=hw)
    colour_scale(ax, 40 + tw + 80 + (2 * hw + 22) / 2, y - 30)
    crop(fig, ax, y - 100, 4000)
    save(fig, "fails")


# Plans: the plan the trained model writes, the plan the original model writes, and what the trained model does.


def mono_width(ax, size):
    a = ax.text(0, 0, "0" * 20, fontsize=size, family=MONO)
    w = a.get_window_extent(ax.figure.canvas.get_renderer()).width / ax.figure.dpi * 100 / 20
    a.remove()
    return w


def plan_text(ax, x, top, text, size, width, wrong):
    """The plan as written, in a monospaced font wrapped to width; layer numbers in `wrong` are red and bold.
    Returns the bottom edge."""
    cw = mono_width(ax, size)
    per = max(int(width // cw), 10)
    step = size * PT * 1.5
    y = top - size * PT * 0.8
    for raw in text.strip().split("\n"):
        words, n, in_layers = [], 0, False
        for wd in raw.split(" "):   # a list written without spaces breaks after its commas
            while len(wd) > per:
                cut = wd.rfind(",", 0, per) + 1 or per
                words.append(wd[:cut])
                wd = wd[cut:]
            words.append(wd)
        rows = [[]]
        for wd in words:
            if n and n + 1 + len(wd) > per:
                rows.append([])
                n = 0
            rows[-1].append(wd)
            n += len(wd) + (1 if n else 0)
        for row in rows:
            cx = x
            for k, wd in enumerate(row):
                if k:
                    cx += cw
                for part in re.findall(r"\d+|[^\d]+", wd):   # digits and the text between them
                    red = in_layers and part.isdigit() and int(part) in wrong
                    ax.text(cx, y, part, fontsize=size, family=MONO, color=RED if red else INK,
                            weight="bold" if red else "normal", ha="left", va="center", zorder=6)
                    cx += cw * len(part)
                if wd == "layers":
                    in_layers = True
                if wd.endswith(";"):
                    in_layers = False
            y -= step
    return y + step - size * PT * 0.8


def layer_bars(ax, x, y, w, h, lo, hi, bars, bar_cols, achieved=None, size_line=None):
    """Per-layer bars (wide, pale) and the achieved change (narrow, orange)."""
    sy = lambda v: y + h * (np.clip(v, lo, hi) - lo) / (hi - lo)
    n = len(bars)
    step = w / n
    if size_line is not None:
        ax.plot([x, x + w], [sy(size_line)] * 2, color=PALE_GRAY, lw=1.2, zorder=2)
        write(ax, x - 12, sy(size_line), f"{size_line:+.0f} SD".replace("-", "−"), 15, color=GRAY, ha="right")
    ax.plot([x, x + w], [sy(0)] * 2, color=GRAY, lw=1.4, zorder=3)
    write(ax, x - 12, sy(0), "0", 15, color=GRAY, ha="right")
    for i in range(n):
        cx = x + (i + 0.5) * step
        if abs(bars[i]) > 1e-6:
            ax.add_patch(Rectangle((cx - step * 0.42, min(sy(0), sy(bars[i]))), step * 0.84, abs(sy(bars[i]) - sy(0)),
                                   fc=bar_cols[i], ec="none", zorder=4))
        if achieved is not None:
            ax.add_patch(Rectangle((cx - step * 0.22, min(sy(0), sy(achieved[i]))), step * 0.44,
                                   abs(sy(achieved[i]) - sy(0)), fc=ACC, ec="none", zorder=5))


def plan():
    """Two held-out layer-pattern instructions followed by the 1.7B model trained on plans alone (r17): the request,
    the plan it writes and what it does, and the plan the original model writes after thinking and what the trained
    model does with that plan (results/plan_examples_r17.pt, from magician/examples.py)."""
    import torch
    recs = {r["idx"]: r for r in torch.load(RES / "plan_examples_r17.pt", weights_only=False)}
    W = 1860
    fig, ax = canvas(W, 5000)
    y = 4960
    write(ax, 24, y - 24, "Plans for layer patterns never seen in training", 32, True, ha="left")
    y -= 90
    tx, tw = 40, 720
    x0, x1 = 880, 1620
    xr = W - 24 - text_width(ax, "pattern match r", 20, True) / 2
    write(ax, (x0 + x1) / 2, y, "change per layer, averaged over reply tokens", 20, True)
    write(ax, xr, y, "pattern match r", 20, True)
    y -= 40
    for idx in (94, 66):
        rec = recs[idx]
        L = rec["own"]["target"].shape[1]
        tgt = rec["own"]["target"].float().numpy()[:, :, 0].mean(0)
        name = rec["own_plan"].split(":")[0]
        text = instruction_only(rec["user"], TASKS[rec["task"]])
        bold = tuple(w + e for w in name.split() if w != "the" for e in ("", "s", "es"))
        th = rich_height(text, 23, W - 120, bold=bold)
        h = th + 64
        rbox(ax, 16, y - h, W - 32, h, PANEL, radius=22, z=1)
        write(ax, 40, y - 26, "INSTRUCTION", 17, True, GRAY, ha="left")
        rich(ax, 40, y - 46, text, 23, W - 120, bold=bold)
        y -= h + 30
        vals = [tgt]
        rows = []
        for key, lab, col in (("own", "plan the trained model writes", ACC),
                              ("orig", "plan the original model writes after thinking", PLANNER)):
            ptxt = rec[f"{key}_plan"]
            got = spec.parse_plan(ptxt, L)
            pv = got[name].mean(1) if name in got else np.zeros(L)
            ach = rec[key]["dz"].float().numpy()[:, :, 0].mean(0)
            wrong = {i for i in range(L) if abs(pv[i] - tgt[i]) > 0.06}
            rows.append((lab, col, ptxt, pv, ach, wrong, rec[key]["r"]))
            vals += [pv, ach]
        lo = min(min(v.min() for v in vals) - 0.4, -1.0)
        hi = max(max(v.max() for v in vals) + 0.4, 1.0)
        size_line = float(tgt[np.argmax(np.abs(tgt))])
        ch = 130
        # requested
        write(ax, tx, y - ch / 2, "requested", 21, True, ha="left")
        layer_bars(ax, x0, y - ch, x1 - x0, ch, lo, hi, tgt, [TGT] * L, size_line=size_line)
        y -= ch + 34
        for lab, col, ptxt, pv, ach, wrong, r in rows:
            ax.scatter([tx + 9], [y - 14], s=200, color=col, linewidths=0, zorder=6)
            write(ax, tx + 28, y - 14, lab, 21, True, ha="left")
            bottom = plan_text(ax, tx, y - 40, ptxt, 17, tw, wrong)
            rh = max(ch, y - bottom + 10)
            cy = y - rh / 2 - 10
            cols = [PALE_RED if i in wrong else TGT for i in range(L)]
            layer_bars(ax, x0, cy - ch / 2, x1 - x0, ch, lo, hi, pv, cols, achieved=ach, size_line=size_line)
            write(ax, xr, cy, f"{r:.2f}".replace("-", "−"), 26, True)
            y -= rh + 34
        write(ax, x0 + (x1 - x0) / (2 * L), y + 12, "0", 15, color=GRAY)
        write(ax, x1 - (x1 - x0) / (2 * L), y + 12, str(L - 1), 15, color=GRAY)
        write(ax, (x0 + x1) / 2, y + 12, "layer", 15, color=GRAY)
        y -= 50
    x = 40
    for lab, kind, col in (("plan layer that matches the request", "bar", TGT),
                           ("plan layer that differs from the request", "bar", PALE_RED),
                           ("change the trained model makes", "bar", ACC)):
        ax.add_patch(Rectangle((x, y - 13), 26, 26, fc=col, ec="none"))
        write(ax, x + 36, y, lab, 19, True, ha="left")
        x += 36 + text_width(ax, lab, 19, True) + 56
    crop(fig, ax, y - 40, 5000)
    save(fig, "plan")


SCORE_ROWS = (("seen", "Kinds of instruction seen in training"), ("new_wording", "Wordings never seen in training"),
              ("new_concepts", "Concepts never seen in training"), ("three_edits", "Three edits in one instruction"),
              ("new_token_patterns", "Token patterns never seen in training"),
              ("new_layer_patterns", "Layer patterns never seen in training"),
              ("all_new", "New concept, layer pattern and wording at once"))


def scores():
    """Each held-out set: the share of the requested change and the pattern match r, original model to trained model
    (mean and 95% bootstrap interval over 96 instructions)."""
    E, _ = data()
    W = 1800
    fig, ax = canvas(W, 2000)
    y = 1960
    write(ax, 24, y - 24, "Qwen3-4B on instructions not used in training", 32, True, ha="left")
    y -= 100
    pct = lambda v: f"{100 * v:.0f}%"
    num = lambda v: f"{v:.2f}".replace("-", "−")
    panels = (("frac", "share of the requested change", 800, 1250, 0.0, 1.0, ((0, "0%"), (0.5, "50%"), (1.0, "100%")),
               pct),
              ("r", "pattern match r", 1370, 1730, -0.2, 0.8, ((0, "0"), (0.4, "0.4"), (0.8, "0.8")), num))
    for _, head, a, b, *_ in panels:
        write(ax, (a + b) / 2, y, head, 21, True)
    y -= 56
    top = y + 26
    for n, (key, label) in enumerate(SCORE_ROWS):
        if key == "new_token_patterns":
            y -= 26
        S = E["sets"][key]["summary"]
        write(ax, 24, y, label, 22, ha="left")
        for k, head, a, b, lo, hi, ticks, fmt in panels:
            sx = lambda v: a + (b - a) * (np.clip(v, lo, hi) - lo) / (hi - lo)
            m, cl, ch = S[k][:3]
            o = S["orig"][k][0]
            ax.plot([sx(o), sx(m)], [y, y], color=PALE_GRAY, lw=3, zorder=2, solid_capstyle="round")
            ax.plot([sx(cl), sx(ch)], [y, y], color=ACC, lw=12, alpha=0.3, zorder=3, solid_capstyle="round")
            ax.scatter([sx(o)], [y], s=200, color=GRAY, zorder=4, linewidths=2, edgecolors="white")
            ax.scatter([sx(m)], [y], s=300, color=ACC, zorder=5, linewidths=2, edgecolors="white")
            if m >= o:
                write(ax, max(sx(ch), sx(m)) + 16, y, fmt(m), 20, True, ha="left")
            else:
                write(ax, min(sx(cl), sx(m)) - 16, y, fmt(m), 20, True, ha="right")
        y -= 70
    for k, head, a, b, lo, hi, ticks, fmt in panels:
        sx = lambda v: a + (b - a) * (np.clip(v, lo, hi) - lo) / (hi - lo)
        for v, lab in ticks:
            ax.plot([sx(v)] * 2, [y + 40, top], color=GRID, lw=1.2, zorder=1)
            write(ax, sx(v), y + 14, lab, 17, color=GRAY)
    y -= 50
    x = 24
    for lab, col, s in (("trained model", ACC, 300), ("original model", GRAY, 200)):
        ax.scatter([x + 10], [y], s=s, color=col, linewidths=0)
        write(ax, x + 30, y, lab, 20, True, ha="left")
        x += 30 + text_width(ax, lab, 20, True) + 60
    ax.plot([x, x + 40], [y, y], color=ACC, lw=12, alpha=0.3, solid_capstyle="round")
    write(ax, x + 56, y, "95% interval", 20, True, ha="left")
    crop(fig, ax, y - 40, 2000)
    save(fig, "scores")


PLAN_GROUPS = (("Qwen3-1.7B", (("n17", "no plan"), ("p17", "plan"), ("r17", "plan, also trained on plan alone"))),
               ("Qwen3-4B", (("p4", "plan"), ("r4", "plan, also trained on plan alone"))))


def base_plan(k, key="new_layer_patterns"):
    """Summary of a set followed from the plans the original model writes after thinking (eval_<arm>think.json:
    evaluate.py --plan_think 1024)."""
    f = RES / f"eval_{k}think.json"
    if f.exists():
        bp = load(f.name)["sets"][key].get("base_plan")
        return bp and bp["summary"]


def dot_legend(ax, x, y, entries, width, size=20):
    """Dots, diamonds or a tick with labels in ink, wrapping to a new line at width; returns the last line's y."""
    for lab, col, kind in entries:
        w = 30 + text_width(ax, lab, size, True)
        if x + w > width - 20:
            x, y = 24, y - 46
        if kind in ("dot", "diamond"):
            ax.scatter([x + 10], [y], s=260 if kind == "dot" else 170, color=col, linewidths=0,
                       marker="o" if kind == "dot" else "D")
        else:
            ax.plot([x + 10, x + 10], [y - 16, y + 16], color=col, lw=3)
        write(ax, x + 30, y, lab, size, True, ha="left")
        x += w + 56
    return y


def plans():
    """Layer patterns never trained on: how often a plan is exactly right, and the pattern match r with the plan the
    trained model writes, the plan the original model writes after thinking, and the exact plan."""
    W = 1800
    fig, ax = canvas(W, 2400)
    y = 2360
    write(ax, 24, y - 24, "Layer patterns never seen in training", 32, True, ha="left")
    y -= 100
    xp, x0, x1, lo, hi = 830, 1000, 1740, -0.1, 0.8
    sx = lambda v: x0 + (x1 - x0) * (np.clip(v, lo, hi) - lo) / (hi - lo)
    write(ax, xp, y, "plan exactly right", 21, True)
    write(ax, (x0 + x1) / 2, y, "pattern match r", 21, True)
    y -= 40
    top = y
    for group, arms in PLAN_GROUPS:
        y -= 26
        write(ax, 24, y, group.upper(), 17, True, GRAY, ha="left")
        y -= 46
        for k, lab in arms:
            if not (RES / f"eval_{k}.json").exists():
                continue
            S = load(f"eval_{k}.json")["sets"]["new_layer_patterns"]
            bp = base_plan(k)
            write(ax, 24, y, lab, 22, ha="left")
            pe = S["summary"].get("plan_exact")
            if pe:
                for dx, col, v in ((-120, ACC, pe[0]), (20, PLANNER, bp and bp["plan_exact"][0])):
                    if v is None:
                        continue
                    ax.scatter([xp + dx], [y], s=150, color=col, linewidths=0)
                    write(ax, xp + dx + 18, y, f"{100 * v:.0f}%", 22, True, ha="left")
            else:
                write(ax, xp, y, "–", 22, True)
            seen = load(f"eval_{k}.json")["sets"]["seen"]["summary"]["r"][0]
            ax.plot([sx(seen)] * 2, [y - 18, y + 18], color=GRAY, lw=3, zorder=3)
            own = S["summary"]["r"][0]
            pts = [(own, ACC, 7)]
            if "forced" in S:
                pts.append((S["forced"]["summary"]["r"][0], GIVEN, 8))
            if bp:
                pts.append((bp["r"][0], PLANNER, 6))
            vs = [p[0] for p in pts]
            ax.plot([sx(min(vs)), sx(max(vs))], [y, y], color=PALE_GRAY, lw=4, zorder=2, solid_capstyle="round")
            for v, col, z in pts:
                ax.scatter([sx(v)], [y], s=300 if col != GIVEN else 190, color=col, zorder=z, linewidths=2,
                           edgecolors="white", marker="o" if col != GIVEN else "D")
            y -= 66
    for v in (0, 0.4, 0.8):
        ax.plot([sx(v)] * 2, [y + 36, top], color=GRID, lw=1.2, zorder=1)
        write(ax, sx(v), y + 12, f"{v:g}", 17, color=GRAY)
    y = dot_legend(ax, 24, y - 50, (("plan the trained model writes", ACC, "dot"),
                                    ("plan the original model writes after thinking", PLANNER, "dot"),
                                    ("exact plan", GIVEN, "diamond"), ("patterns seen in training", GRAY, "tick")), W)
    crop(fig, ax, y - 40, 2400)
    save(fig, "plans")


RULES = (("every45", "every 4th or 5th layer"), ("power2", "powers of two"), ("prime", "prime-numbered layers"),
         ("square", "perfect squares"), ("except", "every layer except"), ("vshape", "V shape"), ("wave", "wave"))


def rules(arm="r17", planner_arm="r17think"):
    """Layer patterns never trained on, by rule (the first layer pattern of each instruction): how often each plan is
    exactly right, and the pattern match r with the plan the trained model writes, the plan the original model writes
    after thinking, and the exact plan."""
    S = load(f"eval_{arm}.json")["sets"]["new_layer_patterns"]
    B = load(f"eval_{planner_arm}.json")["sets"]["new_layer_patterns"]["base_plan"]["items"]
    own, given = S["items"], S["forced"]["items"]
    W = 1800
    fig, ax = canvas(W, 2400)
    y = 2360
    write(ax, 24, y - 24, "Layer patterns never seen in training, by rule (Qwen3-1.7B)", 32, True, ha="left")
    y -= 100
    xp, x0, x1, lo, hi = 640, 820, 1740, -0.3, 0.8
    sx = lambda v: x0 + (x1 - x0) * (np.clip(v, lo, hi) - lo) / (hi - lo)
    write(ax, xp, y, "plan exactly right", 21, True)
    write(ax, (x0 + x1) / 2, y, "pattern match r", 21, True)
    y -= 66
    top = y + 30
    for fam, lab in RULES:
        J = [j for j, it in enumerate(own) if it["fams"][0] == fam]
        if not J:
            continue
        m = lambda items, k: float(np.mean([items[j].get(k, 0.0) for j in J]))
        write(ax, 24, y, lab, 22, ha="left")
        for dx, col, v in ((-120, ACC, m(own, "plan_exact")), (20, PLANNER, m(B, "plan_exact"))):
            ax.scatter([xp + dx], [y], s=150, color=col, linewidths=0)
            write(ax, xp + dx + 18, y, f"{100 * v:.0f}%", 22, True, ha="left")
        pts = [(m(own, "r"), ACC, 7), (m(B, "r"), PLANNER, 6), (m(given, "r"), GIVEN, 8)]
        vs = [p[0] for p in pts]
        ax.plot([sx(min(vs)), sx(max(vs))], [y, y], color=PALE_GRAY, lw=4, zorder=2, solid_capstyle="round")
        for v, col, z in pts:
            ax.scatter([sx(v)], [y], s=300 if col != GIVEN else 190, color=col, zorder=z, linewidths=2,
                       edgecolors="white", marker="o" if col != GIVEN else "D")
        y -= 66
    for v in (0, 0.4, 0.8):
        ax.plot([sx(v)] * 2, [y + 36, top], color=GRID, lw=1.2, zorder=1)
        write(ax, sx(v), y + 12, f"{v:g}", 17, color=GRAY)
    y = dot_legend(ax, 24, y - 50, (("plan the trained model writes", ACC, "dot"),
                                    ("plan the original model writes after thinking", PLANNER, "dot"),
                                    ("exact plan", GIVEN, "diamond")), W)
    crop(fig, ax, y - 40, 2400)
    save(fig, "rules")


FIGURES = ("hero", "scores", "fails", "plan", "plans", "rules")

if __name__ == "__main__":
    style()
    TASKS = json.loads((ROOT / "magician" / "tasks.json").read_text())
    for n in sys.argv[1:] or FIGURES:
        globals()[n]()
