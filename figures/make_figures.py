"""Draw every figure in figures/ from results/: eval_r4.json and profiles_r4.pt of the evaluated adapter,
step_example_r4.pt and projection_example.pt. Usage: python make_figures.py [figure ...] (default: every figure)."""

import json
import random
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
ASSETS = OUT / "assets"
sys.path.insert(0, str(ROOT / "magician"))
import concepts as K  # noqa: E402
import intervals  # noqa: E402
import spec  # noqa: E402

INK = "#1D1D1F"
MUTED = "#6E6E73"
PANEL = "#F4F4F6"
LAVENDER = "#F2ECFE"     # the trained model's panels
PURPLE = "#7B3FE4"       # the trained model
PURPLE_LIGHT = "#E3D6FB"
ORIG = "#A1A1A6"         # the original model
NEW = "#FFE27A"          # highlight: held out from training
NEW_PANEL = "#FFF7D6"
BLUE = "#3478F6"         # the user's message
REPLY = "#E9E9EB"        # the model's message
TGT = "#D1D1D6"          # the requested change, as bars
RED = "#D2432F"
MONO = "Menlo"
ARM = "r4"               # the Qwen3-4B adapter of every figure: writes a plan in its thinking block
W = 1200                 # figure width in canvas units (0.01 inch; 2400 pixels at 200 dpi)

PT = 100 / 72  # canvas units per typographic point
TASKS = None   # the questions (magician/tasks.json), set in main


def style():
    installed = {f.name for f in font_manager.fontManager.ttflist}
    families = [f for f in ("Helvetica Neue", "Helvetica", "Arial") if f in installed] + ["DejaVu Sans"]
    plt.rcParams.update({"font.family": families, "pdf.fonttype": 42, "svg.fonttype": "none"})


def load(name):
    return json.loads((RES / name).read_text())


def save(fig, name):
    for problem in layout_problems(fig):
        print(f"{name}: {problem}")
    fig.savefig(OUT / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("wrote", name)


def layout_problems(fig, tol=2.0):
    """Text that leaves the canvas, overlaps other text, or is crossed by a line or an arrow (display pixels)."""
    from matplotlib.path import Path
    from matplotlib.transforms import Bbox
    r = fig.canvas.get_renderer()
    ax = fig.axes[0]
    texts = [(t.get_text().replace("\n", " "), t.get_window_extent(r)) for t in ax.texts if t.get_text().strip()]
    out = []
    F = fig.bbox
    for t, b in texts:
        if b.x0 < F.x0 - tol or b.x1 > F.x1 + tol or b.y0 < F.y0 - tol or b.y1 > F.y1 + tol:
            out.append(f"cut off: {t[:50]!r}")
    for i, (t, b) in enumerate(texts):
        for u, c in texts[i + 1:]:
            box = Bbox.intersection(b, c)
            if box is not None and box.width > tol and box.height > 0.3 * min(b.height, c.height):
                out.append(f"overlap: {t[:40]!r} / {u[:40]!r}")
    drawn = [ln.get_transform().transform_path(ln.get_path()) for ln in ax.lines if ln.get_gid() != "strike"]
    drawn += [p.get_transform().transform_path(p.get_path()) for p in ax.patches if isinstance(p, FancyArrowPatch)]
    paths = [Path(v) for path in drawn for v in path.to_polygons(closed_only=False) if len(v) > 1]
    boxes = [p.get_window_extent(r) for p in ax.patches if isinstance(p, FancyBboxPatch)]
    bars = [p.get_window_extent(r) for p in ax.patches if type(p) is Rectangle]
    for t, b in texts:
        inner = Bbox.from_extents(b.x0 + tol, b.y0 + tol, b.x1 - tol, b.y1 - tol)
        if any(path.intersects_bbox(inner, filled=False) for path in paths):
            out.append(f"line through: {t[:50]!r}")
        for c in boxes:
            box = Bbox.intersection(b, c)
            if box is not None and box.width > tol and box.height > 0.3 * b.height and not (
                    b.x0 >= c.x0 - tol and b.x1 <= c.x1 + tol and b.y0 >= c.y0 - tol and b.y1 <= c.y1 + tol):
                out.append(f"crosses a box edge: {t[:50]!r}")
                break
        ink = Bbox.from_extents(b.x0, b.y0 + 0.2 * b.height, b.x1, b.y1 - 0.2 * b.height)
        if any((x := Bbox.intersection(ink, c)) is not None and x.width > tol and x.height > tol for c in bars):
            out.append(f"on a bar: {t[:50]!r}")
    return out


# Canvas primitives (units of 0.01 inch).

def canvas(height, width=W):
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=200, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def crop(fig, ax, bottom, top):
    width = ax.get_xlim()[1]
    ax.set_ylim(bottom, top)
    fig.set_size_inches(width / 100, (top - bottom) / 100)


def rbox(ax, x, y, w, h, fill, radius=14, z=2, edge="none", lw=0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
                                edgecolor=edge, linewidth=lw, linestyle=ls, zorder=z))


def write(ax, x, y, text, size=28, bold=False, color=INK, ha="center", va="center", **kw):
    return ax.text(x, y, text, fontsize=size, weight="bold" if bold else "normal", color=color,
                   ha=ha, va=va, zorder=kw.pop("zorder", 6), **kw)


def text_width(ax, text, size, bold=False, family=None):
    artist = ax.text(0, 0, text, fontsize=size, weight="bold" if bold else "normal", family=family)
    box = artist.get_window_extent(ax.figure.canvas.get_renderer())
    artist.remove()
    return box.width / ax.figure.dpi * 100


def image(ax, path, cx, cy, size, z=5):
    ax.imshow(plt.imread(path), extent=(cx - size / 2, cx + size / 2, cy - size / 2, cy + size / 2), zorder=z)


def arrow(ax, a, b, color=INK, lw=3, rad=0.0, z=4, head=22):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=f"-|>,head_length={head / 20},head_width={head / 34}",
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=z,
                                 mutation_scale=20, shrinkA=0, shrinkB=0))


def number(ax, x, y, k, size=26):
    ax.add_patch(Circle((x, y), size * 0.95, color=INK, zorder=5))
    write(ax, x, y - 1, str(k), size, True, "white")


def pill(ax, x, y, text, size, fill, color=INK, bold=True):
    """A rounded label whose left edge is x and centre height y; returns its right edge."""
    w = text_width(ax, text, size, bold) + size * 1.1
    h = size * PT * 1.55
    rbox(ax, x, y - h / 2, w, h, fill, radius=h / 2, z=4)
    write(ax, x + w / 2, y, text, size, bold, color)
    return x + w


# Text with bold and highlighted words, wrapped to a width.

WEAK_ENDS = ("it", "one", "a", "the", "of", "to", "and", "on", "in", "your", "by")


def curly(text):
    text = re.sub(r"(\w)'(\w)", "\\1\u2019\\2", text)
    text = re.sub(r'(^|[\s(\[])"', "\\1\u201c", text).replace('"', "\u201d")
    return re.sub(r"'(?=\w)", "\u2018", text).replace("'", "\u2019")


def rich(ax, x, top, text, size, width, marks=(), color=INK, leading=1.5, max_lines=None):
    """Text from (x, top), wrapped to width. marks: (start, end, kind) character spans, kind 'bold', 'new' (bold on a
    highlight: held out from training) or 'strike' (faded and struck through: removed). A line never ends on a short
    function word. Returns the bottom edge."""
    text = curly(text)
    space = text_width(ax, "a a", size) - text_width(ax, "aa", size)
    words = []
    for m in re.finditer(r"\S+", text):
        kinds = {k for a, b, k in marks if a < m.end() and b > m.start()}
        new, bold = ("strike" if "strike" in kinds else "new" in kinds), bool(kinds - {"strike"})
        words.append((m.group(), bold, new, text_width(ax, m.group(), size, bold)))
    lines, line, used = [], [], 0
    for wd in words:
        if line and used + space + wd[3] > width:
            moved = []
            while len(line) > 1 and line[-1][0].lower() in WEAK_ENDS and not line[-1][1]:
                moved.insert(0, line.pop())
            lines.append(line)
            line = moved
            used = sum(m[3] for m in moved) + space * max(len(moved) - 1, 0)
        used += (space if line else 0) + wd[3]
        line.append(wd)
    lines.append(line)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and sum(w[3] for w in last) + space * len(last) + text_width(ax, "…", size) > width:
            last.pop()
        if last:
            t, b, n, _ = last[-1]
            last[-1] = (t.rstrip(".,;:") + "…", b, n, text_width(ax, t.rstrip(".,;:") + "…", size, b))
    step = size * PT * leading
    for i, ln in enumerate(lines):
        cy = top - size * PT * 0.8 - i * step
        segs = []   # runs of words with the same style: [bold, new, text]
        for t, b, n, _ in ln:
            if segs and segs[-1][0] == b and segs[-1][1] == n:
                segs[-1][2] += " " + t
            else:
                segs.append([b, n, t])
        cx = x
        for k, (b, n, t) in enumerate(segs):
            if k:
                cx += space
            tw = text_width(ax, t, size, b)
            if n == "strike":
                write(ax, cx, cy, t, size, b, color, ha="left", alpha=0.45)
                ax.plot([cx - 3, cx + tw + 3], [cy + 1, cy + 1], color=color, lw=2.5, alpha=0.8, zorder=6, gid="strike")
                cx += tw
                continue
            if n:
                hh = size * PT * 1.3
                rbox(ax, cx - 5, cy - hh / 2 - 2, tw + 10, hh, NEW, radius=8, z=3)
            write(ax, cx, cy, t, size, b, INK if n else color, ha="left")
            cx += tw
    return top - size * PT * 0.8 - (len(lines) - 1) * step - size * PT * 0.7


def rich_height(text, size, width, marks=(), **kw):
    fig, ax = canvas(2000)
    h = 2000 - rich(ax, 0, 2000, text, size, width, marks, **kw)
    plt.close(fig)
    return h


def bubble(ax, x, top, w, text, fill, color, size, marks=(), tail="right", pad=26, max_lines=None):
    """A chat bubble whose top-left corner is (x, top); returns its bottom edge."""
    h = 2 * pad + rich_height(text, size, w - 2 * pad, marks, max_lines=max_lines)
    rbox(ax, x, top - h, w, h, fill, radius=30, z=3)
    side = 1 if tail == "right" else -1
    edge = x + w if tail == "right" else x
    tip = [(edge - side * 34, top - h), (edge + side * 14, top - h - 6), (edge - side * 4, top - h + 30)]
    ax.add_patch(Polygon(tip, closed=True, fc=fill, ec="none", zorder=3))
    rich(ax, x + pad, top - pad, text, size, w - 2 * pad, marks, color=color, max_lines=max_lines)
    return top - h


# What an instruction says, and which of its parts are held out from training.

NEW_LAYER_PHRASES = [r"on prime-numbered layers", r"on every layer whose number is prime",
                     r"on layers whose number is a perfect square", r"on the square-numbered layers",
                     r"on layers whose number is a power of two", r"on layers 1, 2, 4, 8 and so on, doubling each time",
                     r"in every layer except layers \d+ through \d+", r"everywhere but layers \d+ to \d+",
                     r"on every (fourth|fifth) layer starting at layer \d+",
                     r"every [45] layers, starting from layer \d+",
                     r"on layers whose number is divisible by [45]", r"in a V shape[^.;]*", r"more the farther[^.;]*",
                     r"in a (smooth )?wave[^.;]*"]
NEW_TOKEN_PHRASES = [r"on every other token of your reply, starting with the first",
                     r"on alternating tokens of your reply",
                     r"less and less as your reply goes on", r"fading out over the course of your reply",
                     r"only between tokens \d+ and \d+ of your reply", r"on reply tokens \d+ through \d+"]
NEW_WORDING = ([re.escape(v) for v in spec.UP["test"] + spec.DOWN["test"]]
               + [r"the activation for", r"\bfeature\b", r"your inner sense of", r"Quietly, inside your layers,",
                  r"Your hidden-state assignment for this reply:", r"Before you answer and the whole time you answer,",
                  r"Instruction for your residual stream:", r"Secret request:", r"Do not talk about it\."])


def markdown(text):
    """Text with markdown's ** markers removed, and bold marks where they enclosed text."""
    parts = " ".join(text.split()).split("**")
    out, marks, pos = "", [], 0
    for k, part in enumerate(parts):
        if k % 2 and k < len(parts) - 1:
            marks.append((pos, pos + len(part), "bold"))
        out += part
        pos += len(part)
    return out, marks


def instruction_only(user, task):
    """The instruction without the question."""
    return " ".join(user.replace(task, "").split())


def spans(text, patterns, kind):
    return [(m.start(), m.end(), kind) for p in patterns for m in re.finditer(p, text)]


def concept_patterns(cis):
    out = []
    for ci in cis:
        for w in {K.NAMES[ci], K.PHRASES[ci]}:
            w = re.sub(r"^the ", "", w)
            out.append(r"\b" + re.escape(w) + r"s?\b")
    return out


def marks_for(text, it, new):
    """Bold concept names; highlight what training never used: new in {'concept', 'wording', 'layers', 'tokens'}."""
    out = spans(text, concept_patterns(it["concepts"]), "new" if "concept" in new else "bold")
    if "wording" in new:
        out += spans(text, NEW_WORDING, "new")
    if "layers" in new:
        out += spans(text, NEW_LAYER_PHRASES, "new")
    if "tokens" in new:
        out += spans(text, NEW_TOKEN_PHRASES, "new")
    return out


# Requested and achieved changes as heatmaps: layers upward, reply tokens rightward.

DIV = LinearSegmentedColormap.from_list("lowered_raised", [(0, "#0D366B"), (0.2, "#2A78D6"), (0.38, "#9EC5F4"),
                                                           (0.5, "#EDEDEA"), (0.62, "#F7B19F"), (0.8, "#E34948"),
                                                           (1, "#7D1717")])
DIV.set_bad((0, 0, 0, 0))
LIM = 3.0   # SD at either end of the colour scale
T_SHOWN = 64


def data(arm=ARM):
    import torch
    return load(f"eval_{arm}.json"), torch.load(RES / f"profiles_{arm}.pt", weights_only=False)


def typical(items, keep=lambda it: True):
    """Index of the instruction whose pattern match r is closest to the median of the kept instructions."""
    J = [j for j, it in enumerate(items) if keep(it)]
    med = float(np.median([items[j]["r"] for j in J]))
    return min(J, key=lambda j: abs(items[j]["r"] - med))


def grid(prof, key, k=0):
    """[T_SHOWN, L] change of the k-th requested concept; tokens after the end of the reply are NaN."""
    a = prof[key].float().numpy()[:, :, k]
    out = np.full((T_SHOWN, a.shape[1]), np.nan)
    out[:min(len(a), T_SHOWN)] = a[:T_SHOWN]
    return out


def heat(ax, x, y, w, h, A, radius=10):
    rbox(ax, x, y, w, h, "white", radius=radius, z=2)
    im = ax.imshow(np.clip(A.T, -LIM, LIM), cmap=DIV, vmin=-LIM, vmax=LIM, origin="lower", interpolation="nearest",
                   extent=(x, x + w, y, y + h), zorder=3, aspect="auto")
    clip = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}", transform=ax.transData,
                          facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)


def heat_axes(ax, x, y, w, h, L, size=22, tokens=True, layers=True):
    if layers:
        write(ax, x - 10, y + 10, "0", size, color=MUTED, ha="right")
        write(ax, x - 10, y + h - 10, str(L - 1), size, color=MUTED, ha="right")
        write(ax, x - 24, y + h / 2, "layer", size, color=MUTED, rotation=90)
    if tokens:
        write(ax, x + 4, y - 22, "0", size, color=MUTED, ha="left")
        write(ax, x + w / 2, y - 22, "reply token", size, color=MUTED)
        write(ax, x + w - 4, y - 22, str(T_SHOWN - 1), size, color=MUTED, ha="right")


def colour_scale(ax, cx, y, w=440, h=26, size=26):
    """The colour scale, centred at cx: lowered on the left, raised on the right."""
    x = cx - w / 2
    ramp = np.linspace(-LIM, LIM, 256)[None, :]
    im = ax.imshow(ramp, cmap=DIV, vmin=-LIM, vmax=LIM, extent=(x, x + w, y - h / 2, y + h / 2), zorder=3,
                   aspect="auto")
    clip = FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=8", transform=ax.transData,
                          facecolor="none", edgecolor="none")
    ax.add_patch(clip)
    im.set_clip_path(clip)
    write(ax, x - 18, y, "lowered", size, True, ha="right")
    write(ax, x + w + 18, y, "raised", size, True, ha="left")
    for v, lab in ((-LIM, f"−{LIM:.0f} SD"), (0, "0"), (LIM, f"+{LIM:.0f} SD")):
        write(ax, x + w * (v + LIM) / (2 * LIM), y - h / 2 - 26, lab, size - 4, color=MUTED)


def block(target):
    """(l0, l1, t0, t1) of a target that is one value on a block of layers and reply tokens, else None."""
    nz = np.abs(target) > 1e-6
    if not nz.any():
        return None
    ls, ts = np.nonzero(nz.any(0))[0], np.nonzero(nz.any(1))[0]
    sub = target[ts[0]:ts[-1] + 1, ls[0]:ls[-1] + 1]
    return (ls[0], ls[-1], ts[0], ts[-1]) if np.allclose(sub, sub.flat[0]) else None


def requested_box(ax, x, y, w, h, b, L, T):
    l0, l1, t0, t1 = b
    bx, by = x + w * t0 / T, y + h * l0 / L
    rbox(ax, bx, by, w * (t1 + 1 - t0) / T, h * (l1 + 1 - l0) / L, "none", radius=6, z=5, edge=INK, lw=3,
         ls=(0, (4, 3)))


def requested_ticks(ax, x, y, w, h, tgt, L):
    """Marks right of a heatmap at the layers where the target is nonzero."""
    on = np.abs(np.nan_to_num(tgt)).max(0) > 1e-6
    for layer in np.nonzero(on)[0]:
        rbox(ax, x + w + 8, y + h * layer / L + 1, 18, h / L - 2, INK, radius=2, z=4)


# 1. The trained model next to the original model on one instruction.

def overview(s="all_new", i=65):
    """An instruction whose concept, wording and layer pattern are all held out from training (one of the best of its
    test set), the trained model's plan in its thinking block and its reply, and the concept's change per layer and
    reply token for the original Qwen3-4B and the trained model (results/eval_r4.json, profiles_r4.pt)."""
    E, P = data()
    it, prof = E["sets"][s]["items"][i], P[s][i]
    name = K.NAMES[it["concepts"][0]]
    fig, ax = canvas(3600)
    y = 3560
    text = " ".join(it["user"].split())
    y = bubble(ax, 140, y, W - 170, text, BLUE, "white", 30, marks_for(text, it, ("concept", "wording", "layers")))
    y -= 44
    lab = "Concept, wording, and layer pattern never seen in training"
    pill(ax, W - 30 - text_width(ax, lab, 24, True) - 27, y, lab, 24, NEW)
    y -= 50
    pw, gap = (W - 3 * 24) / 2, 24
    tgt = grid(prof, "target")
    b = block(tgt)
    L = tgt.shape[1]
    top = y
    panels = (("Qwen3-4B", "robot.png", it["orig_reply"], "orig_dz", PANEL, INK),
              ("Trained", "magician.png", it["reply"], "dz", LAVENDER, PURPLE))
    ends = []
    for k, (head, avatar, reply, key, fill, hc) in enumerate(panels):
        px = 24 + k * (pw + gap)
        yy = top - 70
        write(ax, px + pw / 2, yy, head, 46, True, hc)
        yy -= 60
        yy = thinking_block(ax, px + 30, yy, pw - 60, it["plan"] if key == "dz" else None, False, size=20,
                            wrong=plan_diff(it, prof)[0]) - 30
        asz = 120 if avatar == "robot.png" else 175
        rt, rm = markdown(reply)
        ry = bubble(ax, px + 160, yy, pw - 190, rt, "white", INK, 25, rm, tail="left",
                    max_lines=4 if key == "dz" else 7)
        image(ax, ASSETS / avatar, px + 80, ry + 40 + (12 if avatar == "magician.png" else 0), asz, z=6)
        ends.append(ry)
    yy = min(ends) - 60
    hw = pw - 120
    hh = hw * 36 / 64
    for k, (head, avatar, reply, key, fill, hc) in enumerate(panels):
        px = 24 + k * (pw + gap)
        lab = f"change in {name} representation"
        write(ax, px + 30, yy, lab, fit(ax, lab, 26, pw - 60, True), True, ha="left")
        heat(ax, px + 80, yy - 34 - hh, hw, hh, grid(prof, key))
        if b:
            requested_box(ax, px + 80, yy - 34 - hh, hw, hh, b, L, T_SHOWN)
        else:
            requested_ticks(ax, px + 80, yy - 34 - hh, hw, hh, tgt, L)
        heat_axes(ax, px + 80, yy - 34 - hh, hw, hh, L)
    bottom = yy - 34 - hh - 60
    for k, (*_, fill, _) in enumerate(panels):
        rbox(ax, 24 + k * (pw + gap), bottom, pw, top - bottom, fill, radius=30, z=1)
    y = bottom - 60
    colour_scale(ax, W / 2, y)
    y -= 100
    vals = sorted({round(float(v), 2) for v in tgt[~np.isnan(tgt)] if abs(v) > 1e-6})
    if b:
        lab = f"requested change: {tgt[b[2], b[0]]:+.0f} SD"
        x = W / 2 - (76 + text_width(ax, lab, 26, True)) / 2
        rbox(ax, x, y - 18, 56, 36, "none", radius=6, z=5, edge=INK, lw=3, ls=(0, (4, 3)))
    else:
        lab = f"requested change: {vals[0]:+g} SD on the marked layers".replace("-", "−")
        x = W / 2 - (76 + text_width(ax, lab, 26, True)) / 2
        rbox(ax, x + 20, y - 6, 18, 12, INK, radius=2, z=4)
    write(ax, x + 76, y, lab, 26, True, ha="left")
    crop(fig, ax, y - 50, 3600)
    save(fig, "overview")


# 2. How a concept's representation and an instruction's target are defined.

def fit(ax, text, size, width, bold=False):
    """The largest font size up to size at which text fits in width."""
    while size > 12 and text_width(ax, text, size, bold) > width:
        size -= 1
    return size


def projection(ax, x, top, ci_name):
    """The representation at one layer and reply token, drawn from x: the residual stream projected on the concept's
    direction, and that projection in SD over every reply token of the original model's replies to the training
    questions (results/projection_example.pt, from magician/projection_example.py). Returns the bottom edge."""
    import torch
    P = torch.load(RES / "projection_example.pt", weights_only=False)
    z = P["z"].float().numpy()
    write(ax, x + 40, top, f"{ci_name} at layer {P['layer']}", 26, color=MUTED, ha="left")
    # the residual stream vector, projected on the direction
    ox, oy = x + 60, top - 380
    d = np.array([0.93, 0.37])
    d /= np.linalg.norm(d)
    tip = np.array([ox, oy]) + 440 * d
    hv = np.array([ox + 200, oy + 250])
    foot = np.array([ox, oy]) + ((hv - [ox, oy]) @ d) * d
    ax.add_patch(FancyArrowPatch((ox, oy), tuple(tip), arrowstyle="-|>,head_length=1.2,head_width=0.7", color=INK,
                                 lw=5, mutation_scale=20, zorder=4))
    lx = foot[0] + 24
    write(ax, lx, oy + (lx - ox) * d[1] / d[0] - 44, f"{ci_name} direction", 24, True, ha="left")
    ax.add_patch(FancyArrowPatch((ox, oy), tuple(hv), arrowstyle="-|>,head_length=1.2,head_width=0.7", color=BLUE,
                                 lw=5, mutation_scale=20, zorder=5))
    write(ax, hv[0] - 20, hv[1] + 30, "residual stream", 24, True, BLUE, ha="left")
    ax.plot([hv[0], foot[0]], [hv[1], foot[1]], color=MUTED, lw=2.5, ls=(0, (5, 4)), zorder=3)
    off = np.array([d[1], -d[0]]) * 26
    ax.plot([ox + off[0], foot[0] + off[0]], [oy + off[1], foot[1] + off[1]], color=RED, lw=7,
            solid_capstyle="round", zorder=4)
    mid_p = (np.array([ox, oy]) + foot) / 2 + off * 2.6
    write(ax, mid_p[0] + 6, mid_p[1] - 4, "projection", 24, True, RED, ha="left")
    # that projection over every reply token of the original model, in SD
    hx0, hx1, hy0, hh = x + 600, x + 1150, top - 400, 260
    lo, hi = -4.0, float(np.ceil(max(z.max(), P["top_z"]) + 0.5))
    edges = np.linspace(lo, hi, 61)
    cnt, _ = np.histogram(np.clip(z, lo, hi), edges)
    sx = lambda v: hx0 + (hx1 - hx0) * (v - lo) / (hi - lo)
    for c, a, b in zip(cnt, edges[:-1], edges[1:]):
        if c:
            ax.add_patch(Rectangle((sx(a) + 0.5, hy0), sx(b) - sx(a) - 1, hh * c / cnt.max(), fc=TGT, ec="none",
                                   zorder=3))
    ax.plot([hx0, hx1], [hy0, hy0], color=INK, lw=2, zorder=4)
    for v in range(int(lo), int(hi) + 1, 4):
        ax.plot([sx(v)] * 2, [hy0, hy0 - 10], color=INK, lw=2, zorder=4)
        write(ax, sx(v), hy0 - 32, f"{v:+d} SD".replace("-", "−") if v else "0", 20, color=MUTED)
    tz = P["top_z"]
    ax.plot([sx(tz)] * 2, [hy0, hy0 + hh * 0.75], color=RED, lw=4, zorder=5)
    write(ax, sx(tz) - 10, hy0 + hh * 0.75 + 34, f"“{P['top_token'].strip()}”: {tz:+.1f} SD", 22, True, RED,
          ha="right")
    write(ax, (hx0 + hx1) / 2, hy0 - 76, "z-score on the original model's replies", 22, True, MUTED)
    arrow(ax, (x + 520, top - 260), (hx0 - 20, top - 260), lw=4, head=24)
    return hy0 - 100


def recipe(ci_name="clouds"):
    ci = K.NAMES.index(ci_name)
    fig, ax = canvas(2000, 2 * W)
    y = y1 = 1950
    number(ax, 50, y, 1)
    write(ax, 92, y, "A direction for each concept at each layer", 34, True, ha="left")
    y -= 80
    bw, x2 = 390, 40 + 390 + 70
    boxes = ((40, "average activation on", f"sentences with {ci_name}",
              [s.format(X=K.PHRASES[ci]) for s in K.SENTENCES[:2]], "… 40 sentences"),
             (x2, "the same sentences with", f"{len(K.TRAIN_CONCEPTS)} training concepts",
              [K.SENTENCES[0].format(X="pizza"), K.SENTENCES[1].format(X="pirates")], "… averaged"))
    for x, small, big, sents, more in boxes:
        write(ax, x + bw / 2, y, small, fit(ax, small, 24, bw), color=MUTED)
        write(ax, x + bw / 2, y - 40, big, fit(ax, big, 28, bw, True), True)
        h = 300
        rbox(ax, x, y - 80 - h, bw, h, PANEL, radius=22, z=1)
        yy = y - 104
        for s in sents:
            yy = rich(ax, x + 24, yy, s, 24, bw - 48, spans(s, [re.escape(K.PHRASES[ci]), "pizza", "pirates"], "bold"))
            yy -= 18
        write(ax, x + 24, yy - 16, more, 24, color=MUTED, ha="left")
    mid = y - 80 - 150
    write(ax, 40 + bw + 35, mid, "−", 60, True)
    write(ax, x2 + bw + 35, mid, "=", 60, True)
    ax0 = x2 + bw + 70
    ax.add_patch(FancyArrowPatch((ax0, mid - 90), (ax0 + 150, mid + 90), arrowstyle="-|>,head_length=1.4,"
                                 "head_width=0.8", color=INK, lw=6, mutation_scale=20, zorder=4))
    cx = (ax0 + W - 20) / 2 + 10
    write(ax, cx, mid - 126, ci_name, 28, True)
    write(ax, cx, mid - 170, "direction", 28, True)
    write(ax, cx, mid - 212, "at each layer", 24, color=MUTED)
    number(ax, W + 50, y1, 2)
    write(ax, W + 92, y1, "The representation: a z-scored projection", 34, True, ha="left")
    y = min(projection(ax, W, y1 - 80, ci_name), mid - 212) - 110
    number(ax, 50, y, 3)
    write(ax, 92, y, "An exact target behind every instruction", 34, True, ha="left")
    y -= 60
    L = 36
    tgt = np.zeros((T_SHOWN, L))
    tgt[:, :18] = 1.0
    hw, hh = 380, 380 * 36 / 64
    write(ax, 120 + hw / 2, y - 16, f"target for {ci_name}", 26, True)
    heat(ax, 120, y - 50 - hh, hw, hh, tgt)
    heat_axes(ax, 120, y - 50 - hh, hw, hh, L, size=22)
    write(ax, 120 + hw / 2, y - 50 - 3 * hh / 4, "+1 SD", 28, True)
    write(ax, 120 + hw / 2, y - 50 - hh / 4, "0", 28, True, MUTED)
    wordings = ["Internally, enhance the clouds direction by 1 SD in your shallower half of layers.",
                "Gently boost your internal representation of clouds in layers 0 through 17.",
                "As you write your reply, turn up your clouds activations by 1 standard deviation in the first half "
                "of your layers."]
    x0 = 120 + hw + 110
    arrow(ax, (120 + hw + 20, y - 50 - hh / 2), (x0 - 20, y - 50 - hh / 2), lw=5, head=30)
    bw = (2 * W - 40 - x0 - 2 * 40) / 3
    ends = [bubble(ax, x0 + k * (bw + 40), y - 20, bw, t, BLUE, "white", 24, spans(t, [r"\bclouds\b"], "bold"),
                   pad=22) for k, t in enumerate(wordings)]
    y = min(min(ends), y - 50 - hh - 40)
    crop(fig, ax, y - 40, 2000)
    save(fig, "recipe")


# 3. One training step, in the two forms of a training example.

def training():
    """One training step of the Qwen3-4B adapter (r4) on one training example, in its two forms: after the instruction,
    the exact plan in the thinking block is teacher-forced with a cross-entropy loss; after the bare question, the plan
    is prefilled and loss-masked. The heatmaps are the change the trained adapter makes on the teacher-forced reply
    (results/step_example_r4.pt, from magician/step_example.py); right: each term of the loss, drawn from this
    example, and the backward pass."""
    import torch
    S = torch.load(RES / "step_example_r4.pt", weights_only=False)
    tgt = S["target"].float().numpy()
    n, L = tgt.shape
    name = K.NAMES[S["concept"]]
    question = TASKS[S["task"]]
    instr = instruction_only(S["user"], question)
    prompt = instr + " " + question
    concept = spans(prompt, [re.escape(name)], "bold")
    reply, rmarks = markdown(S["reply"])
    fig, ax = canvas(4000, 2 * W)
    y = 3960
    y -= 10
    pw, pad, gap = 700, 26, 20
    cols = ((16, "Instruction, then plan", concept, "dz", True),
            (16 + pw + gap, "Prefilled plan only", concept + [(0, len(instr), "strike")], "dz_alone", False))
    tops = [rich_height(prompt, 24, pw - 48 - 40 - 2 * pad, c[2]) for c in cols]
    bottoms = []
    for px, title, marks, key, learned in cols:
        x, w = px + 24, pw - 48
        yy = y - 40
        write(ax, x, yy, title, 28, True, ha="left")
        bubble(ax, x + 40, yy - 40, w - 40, prompt, BLUE, "white", 24, marks)
        yy -= 40 + max(tops) + 2 * pad + 40
        write(ax, x + 130, yy, "thinking block", 22, True, MUTED, ha="left")
        yy -= 20
        ph = plan_height(S["plan"], 20, w - 150 - 40)
        bh = ph + 44
        rbox(ax, x + 130, yy - bh, w - 130, bh, LAVENDER if learned else "white", radius=20, z=3,
             edge=PURPLE if learned else MUTED, lw=2.5, ls=(0, (5, 4)))
        plan_text(ax, x + 150, yy - 22, S["plan"], 20, w - 150 - 40, set())
        rbox(ax, x - 8, yy - bh / 2 - 30, 112, 60, INK, radius=14, z=4)
        write(ax, x + 48, yy - bh / 2, "template", 18, True, "white")
        arrow(ax, (x + 108, yy - bh / 2), (x + 126, yy - bh / 2), head=14)
        yy -= bh + 34
        if learned:
            pill(ax, x + 130, yy, "cross-entropy loss", 22, PURPLE, "white")
        else:
            pill(ax, x + 130, yy, "prefilled, loss-masked", 22, REPLY, MUTED)
        yy -= 60
        write(ax, x + 130, yy, "reply, teacher-forced", 22, True, MUTED, ha="left")
        yy = bubble(ax, x + 130, yy - 20, w - 130, reply, "white", INK, 22, rmarks, tail="left", max_lines=2) - 40
        write(ax, x, yy, f"Δz of {name}, LoRA on − off", 22, True, ha="left")
        hx, hw, hh = x + 50, w - 60, 300
        hy = yy - 26 - hh
        A = np.full((T_SHOWN, L), np.nan)
        A[:min(n, T_SHOWN)] = S[key].float().numpy()[:T_SHOWN]
        heat(ax, hx, hy, hw, hh, A)
        requested_box(ax, hx, hy, hw, hh, block(tgt), L, T_SHOWN)
        heat_axes(ax, hx, hy, hw, hh, L, size=20)
        pill(ax, x, hy - 80, "MSE loss to the target", 22, PURPLE, "white")
        bottoms.append(hy - 120)
    yb = min(bottoms)
    for (px, *_), fill in zip(cols, (LAVENDER, PANEL)):
        rbox(ax, px, yb, pw, y - yb, fill, radius=26, z=1)
    # the loss: one term per part of the example, each drawn from this example; then the update
    cx = 16 + 2 * pw + gap + 100
    cw = 2 * W - 16 - cx
    lp, nw = 30, 320
    x0 = cx + lp + nw
    aw = cx + cw - lp - x0
    mw, mh, mg = 112, 66, 44        # one concept's mini heatmaps: width, height, gap between concepts
    ctop = y

    def mini(A, x, top):
        G = np.full((T_SHOWN, L), np.nan)
        G[:min(len(A), T_SHOWN)] = A[:T_SHOWN]
        heat(ax, x, top - mh, mw, mh, G, radius=6)

    def term(top, h, op, name, note=""):
        if op:
            write(ax, cx + lp, top - h / 2, op, 28, True, ha="left")
        write(ax, cx + lp + 40, top - h / 2, name, 28, True, ha="left")
        if note:
            write(ax, cx + lp + 40, top - h / 2 - 34, note, 22, color=MUTED, ha="left", va="top")

    def concepts(top, cols, labels=True):
        """Each concept's Δz above its target; returns the bottom edge."""
        if labels:
            write(ax, x0 - 14, top - mh / 2, "Δz", 22, color=MUTED, ha="right")
            write(ax, x0 - 14, top - 1.5 * mh - 8, "target", 22, color=MUTED, ha="right")
        for k, (dz, tg, nm) in enumerate(cols):
            x = x0 + k * (mw + mg)
            mini(dz, x, top)
            mini(tg, x, top - mh - 8)
            write(ax, x + mw / 2, top - 2 * mh - 22, nm, 21, color=MUTED, va="top")
        return top - 2 * mh - 8 - 64

    write(ax, cx + lp, ctop - lp - 14, "Loss for each example", 28, True, ha="left")
    yy = ctop - lp - 80
    term(yy, 2 * mh + 8, "", "MSE", "requested\nconcept")
    yy = concepts(yy, [(S["dz"].float().numpy(), tgt, name)]) - 30
    term(yy, 2 * mh + 8, "+", "MSE", "other concepts:\nonly their overlap\nwith " + name)
    # the other concepts whose target rises most, changes least, and falls most
    bank, dzb, tgb = list(S["bank"]), S["dz_bank"].float().numpy(), S["target_bank"].float().numpy()
    mean, peak = tgb.mean((0, 1)), np.abs(tgb).max((0, 1))
    mean[bank.index(S["concept"])], peak[bank.index(S["concept"])] = np.nan, np.inf
    picks = (int(np.nanargmax(mean)), int(np.argmin(peak)), int(np.nanargmin(mean)))
    yb2 = concepts(yy, [(dzb[:, :, j], tgb[:, :, j], K.NAMES[bank[j]]) for j in picks], False)
    write(ax, x0 + 3 * mw + 2 * mg + 24, yy - mh - 4, "…", 34, True, MUTED)
    yy = yb2 - 30
    ph = plan_height(S["plan"], 18, aw - 40) + 36
    term(yy, ph, "+", "cross-entropy", "plan tokens,\nafter the\ninstruction")
    rbox(ax, x0, yy - ph, aw, ph, LAVENDER, radius=16, z=3, edge=PURPLE, lw=2.5, ls=(0, (5, 4)))
    plan_text(ax, x0 + 20, yy - 18, S["plan"], 18, aw - 40, set())
    yy -= ph + 110
    nt = S["next_token"]
    bh, gw = 120, aw / 3
    write(ax, x0, yy + 30, "next-token probability", 22, color=MUTED, ha="left")
    term(yy, bh, "+", "KL")
    pmax = max(nt["p_orig"][:3] + nt["p_trained"][:3])
    for k, (tk, po, pt) in enumerate(zip(nt["tokens"][:3], nt["p_orig"], nt["p_trained"])):
        gx = x0 + k * gw + gw / 2
        for d, p, c in ((-23, po, ORIG), (23, pt, PURPLE)):
            rbox(ax, gx + d - 20, yy - bh, 40, bh * p / pmax, c, radius=5, z=3)
        write(ax, gx, yy - bh - 26, tk.strip() or "\\n", 22, family=MONO)
    ax.plot([x0, x0 + aw], [yy - bh] * 2, color=INK, lw=2, zorder=4)
    yy -= bh + 70
    for c, lab in ((ORIG, "original model, question alone"), (PURPLE, "trained model, training example")):
        rbox(ax, x0 - 40, yy - 13, 26, 26, c, radius=5, z=3)
        write(ax, x0 - 2, yy, lab, 22, color=MUTED, ha="left")
        yy -= 40
    cbot = yy - 10
    rbox(ax, cx, cbot, cw, ctop - cbot, PANEL, radius=24, z=1)
    arrow(ax, (16 + 2 * pw + gap + 12, ctop - (ctop - cbot) / 2), (cx - 12, ctop - (ctop - cbot) / 2), lw=5, head=30)
    # one backward pass: the gradient of the loss reaches the LoRA of every layer
    sx, ex = cx + 40, cx + cw - 40
    step = (ex - sx) / L
    bw = step * 0.66
    btop = cbot - 150
    comb = btop + 56
    ax.plot([cx + cw / 2] * 2, [cbot - 6, comb], color=PURPLE, lw=4, zorder=4)
    write(ax, cx + cw / 2 + 24, (cbot + comb) / 2, "backward pass", 26, True, PURPLE, ha="left")
    ax.plot([sx + bw / 2, ex - step + bw / 2], [comb] * 2, color=PURPLE, lw=3, zorder=4, solid_capstyle="round")
    for l in range(L):
        c = sx + l * step + bw / 2
        ax.add_patch(FancyArrowPatch((c, comb), (c, btop + 2), arrowstyle="-|>,head_length=0.45,head_width=0.3",
                                     color=PURPLE, lw=1.8, mutation_scale=20, zorder=4, shrinkA=0, shrinkB=0))
        rbox(ax, sx + l * step, btop - 130, bw, 130, PURPLE_LIGHT, radius=5, z=3, edge=PURPLE, lw=1.6)
    write(ax, sx, btop - 160, "layer 0", 22, color=MUTED, ha="left")
    write(ax, ex - step + bw, btop - 160, f"layer {L - 1}", 22, color=MUTED, ha="right")
    write(ax, cx + cw / 2, btop - 160, "rank-32 LoRA", 24, True, PURPLE)
    y = yb - 60
    colour_scale(ax, 16 + pw - 200, y, w=340, size=24)
    rbox(ax, 16 + pw + 170, y - 16, 44, 32, "none", radius=6, z=5, edge=INK, lw=3, ls=(0, (4, 3)))
    write(ax, 16 + pw + 228, y, "target", 24, True, ha="left")
    crop(fig, ax, min(y - 80, btop - 200), 4000)
    save(fig, "training")


# 4. The training set and the held-out test sets.

LAYER_NAMES = {"all": "all layers", "range": "a range", "first": "first k", "last": "last k", "region": "a third",
               "half": "a half", "single": "one layer", "list": "a list", "parity": "even or odd",
               "every3": "every 3rd", "ramp_up": "ramp up", "ramp_down": "ramp down", "bump": "a bump",
               "every45": "every 4th or 5th", "prime": "prime", "square": "perfect squares", "power2": "powers of two",
               "except": "all but a range", "vshape": "V shape", "wave": "wave", "numset": "number sets",
               "complement": "all but a set", "union": "two sets", "list_long": "a long list",
               "every_m": "every 3rd,\n6th, or 7th", "ramp_part": "partial ramp", "valley": "a valley",
               "blocks": "blocks", "levels": "levels"}
TOKEN_NAMES = {"all": "every token", "first": "first n\ntokens", "after": "after n\ntokens", "grow": "growing",
               "alternate": "alternating", "fade": "fading", "between": "a token span", "window": "a window",
               "every_k": "every 3rd\nor 4th"}


def strip(ax, x, y, w, h, f):
    n = len(f)
    cw = w / n
    for i, v in enumerate(f):
        ax.add_patch(Rectangle((x + i * cw, y), cw * 0.82, h, fc=DIV((np.clip(2 * v, -LIM, LIM) + LIM) / (2 * LIM)),
                               ec="none", zorder=3))


def split():
    """The axes of an instruction, as the r4 run was trained and tested (spec.py): one row per axis, the training
    cell on the left and the held-out cell on the right."""
    fig, ax = canvas(3000, 2 * W)
    y = 2960
    ax0, aw, bx0, bw = 240, 1390, 1660, 720
    write(ax, ax0 + aw / 2, y - 24, "Training set", 34, True)
    lab = "Never seen in training"
    pill(ax, bx0 + bw / 2 - (text_width(ax, lab, 34, True) + 37) / 2, y - 24, lab, 34, NEW)
    y -= 80
    top = y

    def label(text, y0):
        write(ax, 20, y0 - 20, text, 30, True, ha="left", va="top", linespacing=1.2)

    def texts(name, a, b):
        nonlocal y
        label(name, y)
        bottoms = []
        for x0, w, text in ((ax0, aw, a), (bx0, bw, b)):
            top = y - 22
            for part in text.split("\n"):
                top = rich(ax, x0 + 30, top, part, 28, w - 60)
            bottoms.append(top)
        y = min(bottoms) - 44

    def patterns(name, fams_a, fams_b, n, names, pattern):
        nonlocal y
        label(name, y)
        bottoms = []
        for x0, w, fams, per in ((ax0, aw, fams_a, 6), (bx0, bw, fams_b, 2)):
            pitch = (w - 40) / per
            yy, extra = y - 28, 0
            for k, fam in enumerate(fams):
                if k % per == 0 and k:
                    yy -= 104 + extra
                    extra = 0
                cx = x0 + 30 + (k % per) * pitch
                f = pattern(random.Random(3), fam, n)[0]
                strip(ax, cx, yy - 30, pitch - 30, 30, f)
                write(ax, cx, yy - 46, names[fam], 25, ha="left", va="top", linespacing=1.1)
                if "\n" in names[fam]:
                    extra = 34
            bottoms.append(yy - 104 - extra)
        y = min(bottoms) - 6

    texts("Concepts", "135, such as volcanoes, wizards, pirates, thunder, and the moon",
          "45 others, such as robots and chess")
    texts("Wording", "20 verbs (raise, boost, turn down …), 7 sentence forms, and 12 openings",
          "8 other verbs (crank up …),\n3 sentence forms, and 5 openings")
    patterns("Layers", spec.LAYER_TRAIN, spec.LAYER_TEST, 36, LAYER_NAMES, spec.layer_pattern)
    patterns("Reply\ntokens", spec.TOKEN_TRAIN, spec.TOKEN_TEST, 64, TOKEN_NAMES, spec.token_pattern)
    texts("Edits", "1 or 2 concepts per instruction", "3 concepts in one instruction")
    texts("Questions", "232 questions", "78 other questions")
    rbox(ax, bx0 - 10, y + 10, bw + 20, top - y + 4, NEW_PANEL, radius=22, z=1)
    rbox(ax, ax0 - 10, y + 10, aw + 20, top - y + 4, PANEL, radius=22, z=1)
    crop(fig, ax, y - 20, 3000)
    save(fig, "split")


# 5. Scores on every test set.

SCORE_ROWS = (("seen", "Concept, wording, and patterns"), ("new_wording", "Wording"), ("new_concepts", "Concept"),
              ("three_edits", "Three edits at once"), ("new_token_patterns", "Reply-token pattern"),
              ("new_layer_patterns", "Layer pattern"), ("all_new", "Concept, wording, and layer pattern"))


def hbars(ax, x0, x1, lo, hi, y, vals, cols, hs, fmt, cis=None, size=24, gap=6):
    """Horizontal bars from zero for one row; vals, cols, hs per bar, top to bottom. Returns the row's bottom."""
    sx = lambda v: x0 + (x1 - x0) * (np.clip(v, lo, hi) - lo) / (hi - lo)
    yy = y
    for k, (v, c, h) in enumerate(zip(vals, cols, hs)):
        if v is None:
            yy -= h + gap
            continue
        a, b = sorted((sx(0), sx(v)))
        if b - a >= 8:
            rbox(ax, a, yy - h, b - a, h, c, radius=min(6, h / 2, (b - a) / 2), z=3)
        if cis and cis[k]:
            cl, ch = cis[k]
            ax.plot([sx(cl), sx(ch)], [yy - h / 2] * 2, color=INK, lw=2, zorder=4, solid_capstyle="butt")
            for e in (cl, ch):
                ax.plot([sx(e)] * 2, [yy - h / 2 - 7, yy - h / 2 + 7], color=INK, lw=2, zorder=4)
        if fmt:
            end = max(sx(v), sx(cis[k][1]) if cis and cis[k] else sx(v))
            if c == ORIG:   # the original model: always right of zero, so the label stays out of the label column
                write(ax, max(end, sx(0)) + 12, yy - h / 2, fmt(v), size, True, c, ha="left")
            elif v >= 0:
                write(ax, end + 12, yy - h / 2, fmt(v), size, True, c if c != TGT else INK, ha="left")
            else:
                st = min(sx(v), sx(cis[k][0]) if cis and cis[k] else sx(v))
                write(ax, st - 12, yy - h / 2, fmt(v), size, True, c, ha="right")
        yy -= h + gap
    return yy


def axis(ax, x0, x1, lo, hi, y, ticks, top, size=22):
    sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    ax.plot([sx(0)] * 2, [y + 20, top], color=INK, lw=2, zorder=2)
    for v, lab in ticks:
        write(ax, sx(v), y, lab, size, color=MUTED)


def score_block(ax, ox, y, arms, lab_w=390, gap=6):
    """Each test set (SCORE_ROWS), from x offset ox and height y down: per arm (eval json, None or 'orig', colour, bar
    height) the share of the requested change and the pattern match r (median and its exact 95% interval over the
    instructions of the set, intervals.py). Returns the bottom of the axes labels."""
    pct = lambda v: "0%" if round(100 * v) == 0 else f"{100 * v:.0f}%".replace("-", "−")
    dec = lambda v: "0.00" if round(v, 2) == 0 else f"{v:.2f}".replace("-", "−")
    panels = (("share of the", "requested change", ox + 420, ox + 760, 0.0, 1.0, ((0, "0%"), (0.5, "50%"),
                                                                                 (1.0, "100%")), pct, "frac"),
              ("pattern", "match r", ox + 800, ox + 1100, -0.4, 0.8, ((0, "0"), (0.4, "0.4"), (0.8, "0.8")), dec, "r"))
    for h1, h2, a, b, lo, hi, *_ in panels:
        z = a + (b - a) * (0 - lo) / (hi - lo)
        write(ax, z, y - 20, h1, 26, True, ha="left")
        write(ax, z, y - 56, h2, 26, True, ha="left")
    y -= 110
    top = y + 20
    for key, label in SCORE_ROWS:
        if key == "seen":
            pill(ax, ox + 24, y, "In the training set", 24, PANEL)
            y -= 56
        if key == "new_wording":
            y -= 24
            pill(ax, ox + 24, y, "Never seen in training", 24, NEW)
            y -= 56
        rows_bottom = []
        for _, _, a, b, lo, hi, ticks, fmt, k in panels:
            got = []
            for E, sub, _, _ in arms:
                S = E["sets"].get(key, {}).get("summary")
                got.append((S[sub][k] if sub else S[k]) if S else None)
            cis = [g and tuple(g[1:3]) if sub is None else None for g, (_, sub, _, _) in zip(got, arms)]
            rows_bottom.append(hbars(ax, a, b, lo, hi, y, [g and g[0] for g in got], [c for _, _, c, _ in arms],
                                     [h for *_, h in arms], fmt, gap=gap, cis=cis))
        lb = rich(ax, ox + 30, y + 6, label, 27, lab_w - 20)
        y = min(min(rows_bottom), lb) - 22
    for _, _, a, b, lo, hi, ticks, fmt, k in panels:
        axis(ax, a, b, lo, hi, y, ticks, top)
    return y


def legend(ax, x, y, items, size=25):
    """Colour squares with labels in one line, then the 95% interval mark below. Returns the bottom."""
    for lab, col in items:
        rbox(ax, x, y - 15, 30, 30, col, radius=6)
        write(ax, x + 42, y, lab, size, True, ha="left")
        x += 42 + text_width(ax, lab, size, True) + 50
    x, y = 30, y - 56
    ax.plot([x, x + 40], [y, y], color=INK, lw=2)
    for e in (x, x + 40):
        ax.plot([e] * 2, [y - 7, y + 7], color=INK, lw=2)
    write(ax, x + 52, y, "95% interval for the median", size, True, ha="left")
    return y


def scores():
    """Each test set: the share of the requested change and the pattern match r of the trained model and the original
    model (median and its exact 95% interval over 96 instructions; results/eval_r4.json)."""
    E, _ = data()
    fig, ax = canvas(3000)
    y = score_block(ax, 0, 2960, [(E, None, PURPLE, 34), (E, "orig", ORIG, 14)])
    y = legend(ax, 30, y - 70, (("trained model", PURPLE), ("original Qwen3-4B, thinking off", ORIG)))
    crop(fig, ax, y - 40, 3000)
    save(fig, "scores")


TEAL = "#1A9E77"   # trained without plans


def noplan():
    """The same test sets for models trained without plans (no thinking block) next to models trained with plans:
    Qwen3-4B after stage 1 (results/eval_q4c.json) and after stage 3 (eval_r4.json); Qwen3-1.7B trained from the same
    adapter for 2500 steps on the same data without and with plans (eval_n17.json, eval_p17.json)."""
    E4, N4, N17, P17 = (load(f"eval_{a}.json") for a in ("r4", "q4c", "n17", "p17"))
    fig, ax = canvas(3000, 2 * W)
    blocks = ((0, ("Qwen3-4B", "with plans: trained longer, on more layer patterns"),
               [(N4, None, TEAL, 32), (E4, None, PURPLE, 32), (E4, "orig", ORIG, 20)]),
              (W, ("Qwen3-1.7B", "same training data"),
               [(N17, None, TEAL, 32), (P17, None, PURPLE, 32), (P17, "orig", ORIG, 20)]))
    for ox, head, arms in blocks:
        write(ax, ox + 30, 2950, head[0], 34, True, ha="left")
        write(ax, ox + 30, 2900, head[1], 26, color=MUTED, ha="left")
        y = score_block(ax, ox, 2850, arms, gap=8)
    y = legend(ax, 30, y - 70, (("trained without plans", TEAL), ("trained with plans", PURPLE),
                                ("original model, thinking off", ORIG)))
    crop(fig, ax, y - 40, 3000)
    save(fig, "noplan")


# 5b. What the two scores are computed from: every cell of one instruction.

CONCEPT_COLOURS = ("#1A9E77", "#D95F02", "#2A6FDB")
SET_TAGS = {"seen": ("Question never seen in training", ()),
            "new_wording": ("Wording never seen in training", ("wording",)),
            "new_concepts": ("Concept never seen in training", ("concept",)),
            "three_edits": ("Three edits at once, never seen in training", ("three",)),
            "new_layer_patterns": ("Layer pattern never seen in training", ("layers",))}
CELL_SETS = ("new_concepts", "three_edits", "new_layer_patterns")


def datapoints():
    """For three instructions, every (reply token, layer, requested concept) cell as a dot: requested change against
    achieved change of the trained Qwen3-4B, with the least-squares line through the origin; each the instruction
    closest to the median pattern match of its test set (results/profiles_r4.pt)."""
    E, P = data()
    cw, h = 1000, 560
    fig, ax = canvas(3000, cw * len(CELL_SETS))
    texts, cols = [], []
    for c, s in enumerate(CELL_SETS):
        ox, y = c * cw, 2960
        j = typical(E["sets"][s]["items"])
        it, p = E["sets"][s]["items"][j], P[s][j]
        tag, new = SET_TAGS[s]
        pill(ax, ox + 30, y - 22, tag, 24, NEW)
        text = instruction_only(it["user"], TASKS[it["task"]])
        texts.append(rich(ax, ox + 30, y - 66, text, 25, cw - 60, marks_for(text, it, new)))
        cols.append((ox, it, p))
    ly = min(texts) - 46
    heads, stats = [], []
    for ox, it, p in cols:
        q, a = p["target"].float().numpy(), p["dz"].float().numpy()
        cis = list(dict.fromkeys(it["concepts"]))
        lx = ox + 30
        for k, ci in enumerate(cis):
            ax.scatter([lx + 14], [ly], s=180, color=CONCEPT_COLOURS[k], lw=0, zorder=4)
            write(ax, lx + 32, ly, K.NAMES[ci], 25, True, ha="left")
            lx += 32 + text_width(ax, K.NAMES[ci], 25, True) + 40
        slope = float((a * q).sum() / (q ** 2).sum())
        r = float(np.corrcoef(a.ravel(), q.ravel())[0, 1])
        write(ax, ox + 30, ly - 58, f"share of the requested change {100 * slope:.0f}%", 26, True, ha="left")
        write(ax, ox + 30, ly - 102, f"pattern match r {r:.2f}".replace("-", "−"), 26, True, ha="left")
        heads.append(ly - 102)
        stats.append((ox, q, a, cis, slope))
    top = min(heads) - 70
    for ox, q, a, cis, slope in stats:
        x0, x1 = ox + 170, ox + cw - 50
        qlo, qhi = min(q.min(), 0) - 0.4, max(q.max(), 0) + 0.4
        alo, ahi = np.floor(min(a.min(), qlo)), np.ceil(max(a.max(), qhi))
        sx = lambda v: x0 + (x1 - x0) * (v - qlo) / (qhi - qlo)
        sy = lambda v: top - h + h * (v - alo) / (ahi - alo)
        for k in range(len(cis)):
            ax.scatter(sx(q[..., k].ravel()), sy(a[..., k].ravel()), s=16, color=CONCEPT_COLOURS[k], alpha=0.25,
                       lw=0, zorder=3)
        lo, hi = max(qlo, alo), min(qhi, ahi)
        ax.plot([sx(lo), sx(hi)], [sy(lo), sy(hi)], color=MUTED, lw=2.5, ls=(0, (4, 3)), zorder=4)
        ax.plot([sx(qlo), sx(qhi)], [sy(np.clip(slope * qlo, alo, ahi)), sy(np.clip(slope * qhi, alo, ahi))],
                color=INK, lw=3, zorder=5)
        ax.plot([x0, x0], [top - h, top], color=INK, lw=2)
        ax.plot([x0, x1], [top - h, top - h], color=INK, lw=2)
        for v in range(int(np.ceil(qlo)), int(np.floor(qhi)) + 1):
            ax.plot([sx(v)] * 2, [top - h, top - h - 10], color=INK, lw=2)
            write(ax, sx(v), top - h - 32, f"{v:+d}".replace("-", "−") if v else "0", 22, color=MUTED)
        for v in range(int(alo), int(ahi) + 1):
            ax.plot([x0, x0 - 10], [sy(v)] * 2, color=INK, lw=2)
            write(ax, x0 - 16, sy(v), f"{v:+d}".replace("-", "−") if v else "0", 22, color=MUTED, ha="right")
        write(ax, (x0 + x1) / 2, top - h - 76, "requested change (SD)", 24, True)
        write(ax, x0 - 92, top - h / 2, "achieved change (SD)", 24, True, rotation=90)
    y = top - h - 160
    labs = ("slope: share of the requested change", "achieved = requested")
    tw = sum(text_width(ax, t, 26) for t in labs) + 2 * 56 + 70
    x = cw * len(CELL_SETS) / 2 - tw / 2
    for t, col, lw, ls in zip(labs, (INK, MUTED), (3, 2.5), ("-", (0, (4, 3)))):
        ax.plot([x, x + 42], [y, y], color=col, lw=lw, ls=ls)
        write(ax, x + 56, y, t, 26, ha="left")
        x += 56 + text_width(ax, t, 26) + 70
    crop(fig, ax, y - 40, 3000)
    save(fig, "datapoints")


# 6. Typical instructions from the test sets, as heatmaps.

COLUMNS = (("requested", "target", INK), ("trained model", "dz", PURPLE))


def example_text(ax, x, top, w, row, missing_ok=False):
    """The tag, the instruction (parts never seen in training highlighted) and the trained model's thinking block,
    with the parts of its plan that differ from the instruction outlined in red, and (missing_ok) the requested layers
    the plan does not list. Returns the bottom edge."""
    tag, new, it, prof = row
    text = instruction_only(it["user"], TASKS[it["task"]])
    wrong, missing = plan_diff(it, prof)
    assert missing_ok or not missing, missing
    pill(ax, x, top - 40, tag, 26, NEW if new else "white")
    ty = rich(ax, x, top - 84, text, 28, w, marks_for(text, it, new))
    y = thinking_block(ax, x, ty - 24, w, it["plan"], False, size=21, label="trained model's thinking block",
                       wrong=wrong, lsize=22)
    for name, ls in missing.items():
        y = rich(ax, x, y - 20, f"Requested layers not in the plan: {', '.join(map(str, ls))}", 24, w, color=RED)
    return y


def token_axis(ax, x, y, w):
    write(ax, x + 4, y, "0", 22, color=MUTED, ha="left")
    write(ax, x + w / 2, y, "reply token", 22, color=MUTED)
    write(ax, x + w - 4, y, str(T_SHOWN - 1), 22, color=MUTED, ha="right")


def example_grid(ax, top, rows, x=16, width=2 * W - 32, gap=24, missing_ok=False):
    """Panels side by side, one per instruction: text on top, then for each requested concept the requested and the
    trained model's change side by side; the heatmaps of every panel start at the same height. Returns the bottom."""
    pw = (width - (len(rows) - 1) * gap) / len(rows)
    pad, lm, hg = 30, 72, 22
    xs = [x + k * (pw + gap) for k in range(len(rows))]
    hy = min(example_text(ax, px + pad, top, pw - 2 * pad, row, missing_ok) for px, row in zip(xs, rows)) - 50
    bottoms = []
    for px, (_, _, it, prof) in zip(xs, rows):
        hx = px + pad + lm
        hw = (pw - 2 * pad - lm - hg) / 2
        hh = min(hw * 36 / 64, 200)
        yy = hy
        for k, (head, _, col) in enumerate(COLUMNS):
            write(ax, hx + k * (hw + hg) + hw / 2, yy, head, 28, True, col)
        yy -= 34
        for c in range(len(it["concepts"])):
            if len(it["concepts"]) > 1:
                write(ax, hx, yy - 18, K.NAMES[it["concepts"][c]], 26, True, ha="left")
                yy -= 40
            for k, (_, key, _) in enumerate(COLUMNS):
                heat(ax, hx + k * (hw + hg), yy - hh, hw, hh, grid(prof, key, c))
            heat_axes(ax, hx, yy - hh, hw, hh, prof["target"].shape[1], size=22, tokens=False)
            yy -= hh + 14
        for k in range(len(COLUMNS)):
            token_axis(ax, hx + k * (hw + hg), yy - 10, hw)
        bottoms.append(yy - 50)
    bottom = min(bottoms)
    for px in xs:
        rbox(ax, px, bottom, pw, top - bottom, PANEL, radius=26, z=1)
    return bottom


def example_wide(ax, top, row, tw=820, x=16, width=2 * W - 32):
    """One full-width panel: text on the left; on the right one column per requested concept, the requested change
    above the trained model's change. Returns the bottom."""
    _, _, it, prof = row
    pad = 30
    ty = example_text(ax, x + pad, top, tw, row)
    hx = x + pad + tw + 330
    nc = len(it["concepts"])
    hg = 22
    hw = (x + width - pad - hx - (nc - 1) * hg) / nc
    hh = float(np.clip((top - ty - 44 - 30 - 18 - 70) / 2, hw * 36 / 64, hw * 0.85))   # fill the text's height
    L = prof["target"].shape[1]
    yy = top - 44
    for c in range(nc):
        write(ax, hx + c * (hw + hg), yy, K.NAMES[it["concepts"][c]], 26, True, ha="left")
    yy -= 30
    for head, key, col in COLUMNS:
        write(ax, hx - 70, yy - hh / 2, head.replace(" ", "\n"), 28, True, col, ha="right", linespacing=1.1)
        for c in range(nc):
            heat(ax, hx + c * (hw + hg), yy - hh, hw, hh, grid(prof, key, c))
        heat_axes(ax, hx, yy - hh, hw, hh, L, size=22, tokens=False)
        yy -= hh + 18
    for c in range(nc):
        token_axis(ax, hx + c * (hw + hg), yy - 10, hw)
    bottom = min(ty, yy - 50) - 10
    rbox(ax, x, bottom, width, top - bottom, PANEL, radius=26, z=1)
    return bottom


# Instructions shown in examples.png: per test set, one between the 35th and 65th percentile of the set's pattern
# match whose plan is exact and short, chosen for a vivid concept.
EXAMPLE_PICKS = (("seen", 3), ("new_wording", 40), ("new_concepts", 50), ("three_edits", 4))


def examples():
    """Instructions from four test sets near the median pattern match of their set: the requested change and the
    trained model's change, per layer and reply token."""
    E, P = data()
    rows = []
    for s, i in EXAMPLE_PICKS:
        items = E["sets"][s]["items"]
        lo, hi = np.percentile([it["r"] for it in items], [35, 65])
        assert lo <= items[i]["r"] <= hi and items[i]["plan_exact"], (s, i)
        tag, new = SET_TAGS[s]
        rows.append((tag, new, items[i], P[s][i]))
    fig, ax = canvas(4000, 2 * W)
    y = example_grid(ax, 3980, rows[:3]) - 24
    y = example_wide(ax, y, rows[3])
    colour_scale(ax, W, y - 40, size=28)
    crop(fig, ax, y - 110, 4000)
    save(fig, "examples")


def outside():
    """Layer and reply-token patterns never seen in training: for each rule, the instruction closest to the median
    pattern match of that rule."""
    E, P = data()
    rows = []
    for s, fam, tag, new in (("new_layer_patterns", "prime", "Prime-numbered layers, never seen in training",
                              ("layers",)),
                             ("new_layer_patterns", "power2", "Powers of two, never seen in training", ("layers",)),
                             ("new_token_patterns", "tok_alternate", "Alternating tokens, never seen in training",
                              ("tokens",)),
                             ("new_token_patterns", "tok_fade", "Fading over the reply, never seen in training",
                              ("tokens",))):
        items = E["sets"][s]["items"]
        i = typical(items, lambda it: fam in it["fams"])
        rows.append((tag, new, items[i], P[s][i]))
    fig, ax = canvas(4000, 2 * W)
    y = example_grid(ax, 3980, rows[:2]) - 24
    y = example_grid(ax, y, rows[2:])
    colour_scale(ax, W, y - 40, size=28)
    crop(fig, ax, y - 110, 4000)
    save(fig, "outside")


def wave():
    """Waves across layers (never seen in training): the instruction closest to the median pattern match in each of
    the two test sets with waves, and the waves with the highest and the lowest pattern match."""
    E, P = data()
    pool = [(s, j) for s in ("new_layer_patterns", "all_new") for j, it in enumerate(E["sets"][s]["items"])
            if "wave" in it["fams"]]
    new = {"new_layer_patterns": ("layers",), "all_new": ("concept", "wording", "layers")}
    r = lambda sj: E["sets"][sj[0]]["items"][sj[1]]["r"]
    picks = [("new_layer_patterns", typical(E["sets"]["new_layer_patterns"]["items"], lambda it: "wave" in it["fams"]),
              "Wave, median instruction"),
             ("all_new", typical(E["sets"]["all_new"]["items"], lambda it: "wave" in it["fams"]),
              "Wave, concept and wording also new, median"),
             (*max(pool, key=r), "Wave, best instruction"), (*min(pool, key=r), "Wave, worst instruction")]
    rows = [(tag, new[s], E["sets"][s]["items"][j], P[s][j]) for s, j, tag in picks]
    fig, ax = canvas(4000, 2 * W)
    y = example_grid(ax, 3980, rows[:2], missing_ok=True) - 24
    y = example_grid(ax, y, rows[2:], missing_ok=True)
    colour_scale(ax, W, y - 40, size=28)
    crop(fig, ax, y - 110, 4000)
    save(fig, "wave")


def scale():
    """Qwen3-1.7B and Qwen3-4B trained from scratch with the same recipe, without plans (thinking off), evaluated on
    the same test sets (results/eval_s17.json: 3000 steps; eval_s4.json: 2500 steps)."""
    S17, S4 = load("eval_s17.json"), load("eval_s4.json")
    fig, ax = canvas(3000)
    write(ax, 30, 2950, "Same training recipe, no plans", 34, True, ha="left")
    write(ax, 30, 2900, "1.7B trained 3000 steps, 4B trained 2500 steps", 26, color=MUTED, ha="left")
    y = score_block(ax, 0, 2850, [(S17, None, BLUE, 32), (S4, None, PURPLE, 32)], gap=8)
    y = legend(ax, 30, y - 70, (("Qwen3-1.7B", BLUE), ("Qwen3-4B", PURPLE)))
    crop(fig, ax, y - 40, 3000)
    save(fig, "scale")


def jlens():
    """The trained Qwen3-4B's change on each test set read three ways: on the concept vectors used in training, on the
    J-lens (the gradient of the concept's logit with respect to the residual stream), and on random vectors
    (results/jlens_r4.json, from magician/jlens.py)."""
    J = load("jlens_r4.json")
    lens = lambda k: {"sets": {s: {"summary": v[k]} for s, v in J["sets"].items()}}
    fig, ax = canvas(3000)
    y = score_block(ax, 0, 2960, [(lens("trained"), None, PURPLE, 30), (lens("jlens"), None, TEAL, 30),
                                  (lens("random"), None, ORIG, 18)], gap=8)
    y = legend(ax, 30, y - 70, (("concept vectors used in training", PURPLE), ("J-lens", TEAL),
                                ("random vectors", ORIG)))
    crop(fig, ax, y - 40, 3000)
    save(fig, "jlens")


# Zero-shot: original models of five sizes on the test instructions, without training.

ZS_MODELS = (("Qwen3.5-0.8B", 0.8), ("Qwen3.5-2B", 2), ("Qwen3.5-4B", 4), ("Qwen3.5-9B", 9), ("Qwen3.8-27B", 27))


def median_interval(xs, key):
    """Median of the per-instruction scores and its exact 95% interval (intervals.py)."""
    return tuple(intervals.interval(xs, key)[:3])


def zeroshot():
    """The newest Qwen model of each size, not trained, on the test instructions (results/zeroshot/<model>.json):
    share of the requested change and pattern match r over every instruction of every test set, with thinking off and
    with thinking on; the trained Qwen3-4B on the same instructions (results/eval_r4.json) for reference."""
    Z = {m: load(f"zeroshot/{m}.json") for m, _ in ZS_MODELS if (RES / "zeroshot" / f"{m}.json").exists()}
    first = next(iter(Z.values()))
    sets, n = list(first["think_off"]), first["n"]
    E = load("eval_r4.json")
    trained = {k: median_interval([it[k] for s in sets for it in E["sets"][s]["items"][:n]], k)
               for k in ("frac", "r")}
    pct = lambda v: f"{100 * v:.0f}%"
    fig, ax = canvas(1100, 2 * W)
    panels = (("share of the requested change", "frac", ((0, "0%"), (0.25, "25%"), (0.5, "50%"), (0.75, "75%"))),
              ("pattern match r", "r", ((0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"))))
    lo, hi = -0.1, 0.8
    for p, (title, k, ticks) in enumerate(panels):
        x0, x1, y0, y1 = 190 + p * W, (p + 1) * W - 250, 200, 940
        sx = lambda s: x0 + 40 + (x1 - x0 - 80) * np.log(s / 0.8) / np.log(27 / 0.8)
        sy = lambda v: y0 + (y1 - y0) * (v - lo) / (hi - lo)
        ax.plot([x0, x0], [y0, y1], color=INK, lw=2, zorder=2)
        ax.plot([x0, x1], [sy(0)] * 2, color=MUTED, lw=1.5, zorder=1)
        for v, lab in ticks:
            write(ax, x0 - 16, sy(v), lab, 24, color=MUTED, ha="right")
        for _, s in ZS_MODELS:
            write(ax, sx(s), y0 - 34, f"{s:g}B", 24, color=MUTED)
        write(ax, (x0 + x1) / 2, y0 - 84, "parameters, original model", 26, color=MUTED)
        write(ax, x0, y1 + 50, title, 30, True, ha="left")
        for key, col, dx, lab in (("think_off", ORIG, -14, "thinking off"), ("think_on", INK, 14, "thinking on")):
            pts = [(s, median_interval([it[k] for st in Z[m][key].values() for it in st["items"]], k))
                   for m, s in ZS_MODELS if m in Z and Z[m][key]]
            for s, (mu, cl, ch) in pts:
                ax.plot([sx(s) + dx] * 2, [sy(cl), sy(ch)], color=col, lw=2.5, zorder=3)
                ax.scatter([sx(s) + dx], [sy(mu)], s=260, color=col, lw=0, zorder=4)
            if pts:
                s, (mu, _, _) = pts[-1]
                write(ax, sx(s) + dx + 30, sy(mu) + (26 if dx > 0 else -26), lab, 26, True, col, ha="left")
        mu, cl, ch = trained[k]
        ax.plot([sx(4)] * 2, [sy(cl), sy(ch)], color=PURPLE, lw=2.5, zorder=3)
        ax.scatter([sx(4)], [sy(mu)], s=300, color=PURPLE, marker="D", lw=0, zorder=4)
        write(ax, sx(4) + 30, sy(mu), "Qwen3-4B, trained, " + (pct(mu) if k == "frac" else f"{mu:.2f}"), 26, True,
              PURPLE, ha="left")
    crop(fig, ax, 60, 1100)
    save(fig, "zeroshot")


# A plan in a thinking block.

def mono_width(ax, size):
    return text_width(ax, "0" * 20, size, family=MONO) / 20


def plan_text(ax, x, top, text, size, width, wrong=()):
    """The plan as written, in a monospaced font wrapped to width; characters inside the (start, end) spans `wrong`
    are red, bold and outlined. Returns the bottom edge."""
    cw = mono_width(ax, size)
    per = max(int(width // cw), 10)
    step = size * PT * 1.45
    y = top - size * PT * 0.8
    red = np.zeros(len(text) + 1, bool)
    for a, b in wrong:
        red[a:b] = True
    off = 0
    for raw in text.split("\n"):
        words = []   # (index in text, word); a list written without spaces breaks after its commas
        for m in re.finditer(r"\S+", raw):
            at, wd = off + m.start(), m.group()
            while len(wd) > per:
                cut = wd.rfind(",", 0, per) + 1 or per
                words.append((at, wd[:cut]))
                at, wd = at + cut, wd[cut:]
            words.append((at, wd))
        rows, n = [[]], 0
        for at, wd in words:
            if n and n + 1 + len(wd) > per:
                rows.append([])
                n = 0
            rows[-1].append((at, wd))
            n += len(wd) + (1 if n else 0)
        for row in rows:
            cx, runs = x, []   # outlined runs of red characters: [x0, x1]
            for k, (at, wd) in enumerate(row):
                if k:
                    if red[at - 1] and red[at] and runs and abs(runs[-1][1] - cx) < 1e-6:
                        runs[-1][1] += cw   # a space inside a red span
                    cx += cw
                for seg in re.finditer(r"(.)\1*", "".join("1" if red[at + j] else "0" for j in range(len(wd)))):
                    part, r = wd[seg.start():seg.end()], seg.group(1) == "1"
                    ax.text(cx, y, part, fontsize=size, family=MONO, color=RED if r else INK,
                            weight="bold" if r else "normal", ha="left", va="center", zorder=6)
                    if r:
                        if runs and abs(runs[-1][1] - cx) < 1e-6:
                            runs[-1][1] += cw * len(part)
                        else:
                            runs.append([cx, cx + cw * len(part)])
                    cx += cw * len(part)
            hh = size * PT * 1.25
            for x0, x1 in runs:
                rbox(ax, x0 - 1, y - hh / 2, x1 - x0 + 2, hh, "none", radius=6, z=5, edge=RED, lw=2.5)
            y -= step
        off += len(raw) + 1
    return y + step - size * PT * 0.7


def plan_diff(it, prof):
    """Character spans of the trained model's plan that differ from the instruction's exact target: a listed layer
    whose requested value differs (or that the model does not have), a value that differs on every layer it lists,
    a token phrase whose profile differs on the reply tokens, and the line of a concept the instruction did not name.
    Also the requested layers no plan line lists, per concept name."""
    plan, names = it["plan"], [K.NAMES[c] for c in it["concepts"]]
    out, missing, off = [], {}, 0
    for line in plan.split("\n"):
        name, sep, rest = line.partition(": ")
        if name.strip() not in names:
            out.append((off, off + len(line)))
            off += len(line) + 1
            continue
        c = names.index(name.strip())
        tgt = prof["target"][..., c].float().numpy()            # [reply tokens, L]
        L = tgt.shape[1]
        F = tgt[int(np.abs(tgt).sum(1).argmax())]
        g = tgt @ F / max(F @ F, 1e-9)
        listed, at = set(), off + len(name) + len(sep)
        parts = rest.split("; ")
        for p in parts[:-1]:
            m = re.match(r"([+\-−]?[\d.]+) SD on layers (.+)", p)
            if m:
                x = float(m.group(1).replace("−", "-"))
                nums = [(int(q.group()), q.start(), q.end()) for q in re.finditer(r"\d+", m.group(2))]
                bad = [(a, b) for l, a, b in nums if l >= L or abs(F[l] - x) > 0.06]
                listed |= {l for l, _, _ in nums}
                vals = {round(float(F[l]), 1) for l, _, _ in nums if l < L}
                if nums and len(bad) == len(nums) and len(vals) == 1 and 0 not in vals:
                    out.append((at, at + m.end(1)))     # the value is wrong, the layers are right
                else:
                    out += [(at + m.start(2) + a, at + m.start(2) + b) for a, b in bad]
            at += len(p) + 2
        tp = parts[-1]
        try:
            gp = spec.parse_tokens(tp.strip(), spec.T_REPLY)[:len(g)]
            if np.abs(gp - g).max() > 0.06:
                out.append((at, at + len(tp)))
        except (ValueError, IndexError):
            out.append((at, at + len(tp)))
        miss = [l for l in np.nonzero(np.abs(F) > 0.06)[0].tolist() if l not in listed]
        if miss:
            missing[name.strip()] = miss
        off += len(line) + 1
    return out, missing


def thinking_block(ax, x, top, w, text, prefilled, size=22, label="thinking block", wrong=(), lsize=22):
    """A plan inside a thinking block whose top-left corner is (x, top): lavender with a purple dashed edge when the
    trained model generates the plan, white with a grey dashed edge when the plan is prefilled, and empty when text is
    None (the chat template's empty block with thinking turned off). Returns the bottom."""
    write(ax, x, top - lsize * 0.64, label, lsize, True, MUTED, ha="left")
    if prefilled or text is None:
        pill(ax, x + text_width(ax, label, lsize, True) + 16, top - lsize * 0.64, "prefilled" if prefilled else
             "thinking off", lsize - 2, REPLY, MUTED)
    top -= lsize * 1.8
    if text is None:
        rbox(ax, x, top - 56, w, 56, "white", radius=20, z=3, edge=MUTED, lw=2.5, ls=(0, (5, 4)))
        return top - 56
    h = plan_height(text, size, w - 40, wrong) + 44
    rbox(ax, x, top - h, w, h, "white" if prefilled else LAVENDER, radius=20, z=3, edge=MUTED if prefilled else PURPLE,
         lw=2.5, ls=(0, (5, 4)))
    plan_text(ax, x + 20, top - 22, text, size, w - 40, wrong)
    return top - h


def plan_height(text, size, width, wrong=()):
    fig, ax = canvas(2000)
    h = 2000 - plan_text(ax, 0, 2000, text, size, width, wrong)
    plt.close(fig)
    return h


# The Activation Controllability Benchmark (Kowalski et al. 2026) on its own protocol, prompts, concept vectors and
# scoring. Top: the Layer Targeting prompt and, per model, the concept's strength at every layer for each layer the
# prompt asks for. Bottom: every measure for our models next to the paper's 25 models.

# (results key, label, colour); rows without results in results/acb are left out.
ACB_MODELS = (("qwen3_4b", "Qwen3-4B original", INK),
              ("qwen3_4b_magician_noplan", "magician without plans", TEAL),
              ("qwen3_4b_magician", "magician", PURPLE),
              ("qwen3_4b_magician_think", "magician, thinking on", "#3D1A8C"))
# (results key, the benchmark's name, what it asks; the bottom rows in order)
ACB_MEASURES = (("layer_targeting", "Layer Targeting", "the concept only at the layer asked for"),
                ("engage", "Engage", "the concept rises when asked to think about it"),
                ("suppress", "Suppress", "the concept falls when asked not to"),
                ("dial_rank", "Dial", "its strength follows “intensity 1, 2, 3 or 4”"),
                ("temporal_control", "Temporal Control", "only at the start or the end of the sentence"),
                ("coverage", "Coverage", "on every kind of word in the sentence"),
                ("S", "Overall score", "the six measures combined"),
                ("compliance", "Copies the sentence", "the share of answers that match it"))
PAPER_GREY = "#C7C7CC"


def acb_point(C, key):
    """(value, low, high) of S or one of its measures from a SCALAR_CI or CAL_CI json."""
    if key == "S":
        return C["scalar"], C["ci_lo"], C["ci_hi"]
    c = C["components"][f"{key}|proj"]
    return c["obs"], c["lo"], c["hi"]


def tint(colour, f=0.82):
    """colour mixed with white, f of the way to white."""
    rgb = np.array(matplotlib.colors.to_rgb(colour))
    return tuple(rgb + (1 - rgb) * f)


def acb_ridges(ax, x, y, w, d, title, col, labels):
    """One ridge per layer the prompt asks for (bottom: 23, top: 35): the concept's strength at every layer read,
    in SD of the change from the same sentence copied without the instruction, a triangle at the asked-for layer."""
    G = np.array(d["grid_raw"], float)
    asked, reads = d["targets"], d["reads"]
    dy, hs = 64, 13.0
    lx = lambda layer: x + w * (layer + 0.5) / 36
    for i in reversed(range(len(asked))):
        b = y + i * dy
        z = 10 + 3 * (len(asked) - i)
        pts = [(lx(r), b + hs * v) for r, v in zip(reads, G[i])]
        poly = [(lx(reads[0]), b)] + pts + [(lx(reads[-1]), b)]
        ax.add_patch(Polygon(poly, closed=True, fc=tint(col), ec="none", zorder=z))
        ax.plot([x, x + w], [b, b], color=TGT, lw=1.2, zorder=z + 0.5)
        ax.plot(*zip(*pts), color=col, lw=2.6, zorder=z + 1, solid_joinstyle="round")
        ax.scatter([lx(asked[i])], [b + hs * G[i][reads.index(asked[i])]], s=90, color=INK, lw=0, zorder=z + 2)
        if labels:
            write(ax, x - 18, b, str(asked[i]), 22, color=MUTED, ha="right")
    write(ax, x, y + (len(asked) - 1) * dy + 130, title, 30, True, col, ha="left")
    for layer in (0, 12, 24, 35):
        write(ax, lx(layer), y - 44, str(layer), 22, color=MUTED)
    write(ax, x + w / 2, y - 84, "layer", 26, color=MUTED)


def acb_strip(ax, x0, x1, y, vals, ours, key):
    """One measure: the paper's 25 models as grey dots (stacked where they touch), ours as coloured dots with
    calibrated 95% intervals where computed; its own axis below."""
    pts = [v for v in vals if v is not None] + [v for _, v, _, _ in ours] + \
          [b for _, _, b, _ in ours if b is not None and np.isfinite(b[1]) and b[1] < 50]
    lo, hi = min(pts + [0]), max(pts)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    sx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    ax.plot([x0, x1], [y - 34, y - 34], color=INK, lw=1.5, zorder=2)
    if lo < 0 < hi:
        ax.plot([sx(0)] * 2, [y - 34, y + 30], color=TGT, lw=1.5, zorder=1)
    ticks = MaxNLocator(3).tick_values(lo + pad, hi - pad)
    for t in ticks:
        if lo <= t <= hi:
            ax.plot([sx(t)] * 2, [y - 34, y - 42], color=INK, lw=1.5, zorder=2)
            lab = f"{100 * t:g}%" if key == "compliance" else f"{t:g}".replace("-", "−")
            write(ax, sx(t), y - 62, lab, 20, color=MUTED)
    for v in vals:
        if v is not None:
            ax.plot([sx(v)] * 2, [y - 30, y - 6], color=PAPER_GREY, lw=2.5, zorder=3, solid_capstyle="butt")
    for i, (k, v, band, col) in enumerate(ours):
        yy = y + 8 + 22 * i
        if band is not None:
            a, b = band
            ax.plot([sx(max(a, lo)), sx(min(b, hi))], [yy, yy], color=col, lw=3.5, zorder=4, solid_capstyle="butt")
        ax.scatter([sx(v)], [yy], s=230, color=col, lw=0, zorder=5)


def acb():
    """Top: the benchmark's Layer Targeting prompt, and for the original Qwen3-4B and its two magicians the concept's
    strength at every layer for each of the 8 layers a prompt asks for (grid_raw of PAIRED_qwen3_4b.json, from
    magician/acb_checks.py on the benchmark's runs). Bottom: S, the six measures and the share of answers that copy
    the sentence, for our models (SCALAR_CI_<model>.json, compliance.json; intervals from CAL_CI_<model>.json, the
    benchmark's bootstrap calibrated by the double bootstrap, where computed) and for the paper's 25 models
    (paper_models.json, from the benchmark's repository). Thinking off unless a label says otherwise."""
    A = RES / "acb"
    C = {k: json.loads((A / f"SCALAR_CI_{k}.json").read_text()) for k, *_ in ACB_MODELS
         if (A / f"SCALAR_CI_{k}.json").exists()}
    cal = {k: json.loads((A / f"CAL_CI_{k}.json").read_text()) for k in C if (A / f"CAL_CI_{k}.json").exists()}
    comp = json.loads((A / "compliance.json").read_text())
    paper = json.loads((A / "paper_models.json").read_text())["models"]
    P = json.loads((A / "PAIRED_qwen3_4b.json").read_text())
    fig, ax = canvas(4600, 2 * W)

    top = 4560
    prompt = ('Write "The old photograph brought back forgotten memories." exactly. While you write, think about '
              "Volcanoes only at layer 28 of your 36 layers. Don't write anything else.")
    marks = [(prompt.index(s), prompt.index(s) + len(s), "bold") for s in ("Volcanoes", "layer 28")]
    bottom = bubble(ax, 40, top, 2 * W - 80, prompt, BLUE, "white", 27, marks)

    rw, gap, x0 = 620, 100, 170
    ry = bottom - 120 - 7 * 64 - 130
    for p, (k, title, col) in enumerate(ACB_MODELS[:3]):
        acb_ridges(ax, x0 + p * (rw + gap), ry, rw, P[k], title, col, p == 0)
    write(ax, x0 - 18, ry + 7 * 64 + 56, "layer asked for", 22, color=MUTED, ha="left")
    sb = x0 + 3 * rw + 2 * gap + 26
    ax.plot([sb, sb], [ry, ry + 13.0 * 2], color=INK, lw=2.5, zorder=3)
    write(ax, sb + 12, ry + 13, "2 SD", 22, color=INK, ha="left")
    ax.scatter([x0 + 10], [ry - 140], s=90, color=INK, lw=0, zorder=5)
    write(ax, x0 + 30, ry - 140, "the layer asked for", 22, color=MUTED, ha="left")
    write(ax, x0 + 30 + text_width(ax, "the layer asked for", 22) + 60, ry - 140,
          "height: how strongly the concept is present, compared with no instruction", 22, color=MUTED, ha="left")

    # Every measure against the paper's 25 models
    y = ry - 260
    lx0, lx1 = 900, 2330
    items = [("paper", "the paper's 25 models", PAPER_GREY)] + [(k, t, c) for k, t, c in ACB_MODELS if k in C]
    cx = 40
    for k, t, c in items:
        if k == "paper":
            ax.plot([cx + 10] * 2, [y - 12, y + 12], color=c, lw=2.5, zorder=5, solid_capstyle="butt")
        else:
            ax.scatter([cx + 10], [y], s=230, color=c, lw=0, zorder=5)
        write(ax, cx + 30, y, t, 24, True, c if k != "paper" else MUTED, ha="left")
        cx += 30 + text_width(ax, t, 24, True) + 50
    y -= 110
    for key, name, what in ACB_MEASURES:
        vals = [m[key] for m in paper.values()]
        ours = []
        for k, _, col in ACB_MODELS:
            if k not in C:
                continue
            if key == "compliance":
                v, band = comp[k]["main"]["compliant"] / comp[k]["main"]["n"], None
            else:
                v, band = acb_point(C[k], key)[0], (acb_point(cal[k], key)[1:] if k in cal else None)
            ours.append((k, v, band, col))
        write(ax, 40, y + 12, name, 28, True, ha="left")
        write(ax, 40, y - 24, what, 22, color=MUTED, ha="left")
        acb_strip(ax, lx0, lx1, y, vals, ours, key)
        y -= 150
    crop(fig, ax, y + 50, 4590)
    save(fig, "acb")


FIGURES = ("overview", "recipe", "training", "split", "scores", "datapoints", "examples", "outside", "wave",
           "noplan", "scale", "jlens", "zeroshot", "acb")

if __name__ == "__main__":
    style()
    TASKS = json.loads((ROOT / "magician" / "tasks.json").read_text())
    for n in sys.argv[1:] or FIGURES:
        globals()[n]()
