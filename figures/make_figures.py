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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
ASSETS = OUT / "assets"
sys.path.insert(0, str(ROOT / "magician"))
import concepts as K  # noqa: E402
import spec  # noqa: E402

INK = "#1D1D1F"
MUTED = "#6E6E73"
PANEL = "#F4F4F6"
LAVENDER = "#F2ECFE"     # the trained model's panels
PURPLE = "#7B3FE4"       # the trained model
PURPLE_LIGHT = "#E3D6FB"
ORIG = "#A1A1A6"         # the original model
NEW = "#FFE27A"          # highlight: never seen in training
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
    highlight: never seen in training) or 'strike' (faded and struck through: removed). A line never ends on a short
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


# What an instruction says, and which of its parts training never saw.

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
    """Bold concept names; highlight what training never saw: new in {'concept', 'wording', 'layers', 'tokens'}."""
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
    """An instruction whose concept, wording and layer pattern were all never seen in training (one of the best of its
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
    lab = "Training never saw this concept, wording, or layer pattern"
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
        yy = thinking_block(ax, px + 30, yy, pw - 60, it["plan"] if key == "dz" else None, False, size=20) - 30
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
    (results/step_example_r4.pt, from magician/step_example.py); right: the loss and the update."""
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
    number(ax, 50, y - 10, 4)
    write(ax, 92, y - 10, "One training step", 34, True, ha="left")
    y -= 70
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
    # the loss and the update
    cx = 16 + 2 * pw + gap + 110
    cw = 2 * W - 20 - cx
    lp = 30
    lines = [("MSE between Δz and the target over every layer and reply token, with the cells inside and outside the "
              "target weighted equally", 26, True, "white"),
             ("+ MSE for every other training concept, toward the small change the target implies through "
              "overlapping directions", 24, False, "#D8D8DC"),
             ("+ cross-entropy on the plan tokens (left form)", 24, False, "#D8D8DC"),
             ("+ 3 × KL divergence from the original model's next-token probabilities on the question alone", 24, False,
              "#D8D8DC")]
    hs = [rich_height(t, sz, cw - 2 * lp, [(0, len(t), "bold")] if b else ()) for t, sz, b, _ in lines]
    lh = 60 + sum(hs) + 18 * (len(lines) - 1) + 2 * lp
    note = (f"Each step averages this loss over 16 examples; half of the examples with an edit take the right-hand "
            f"form. One backward pass updates the LoRAs of all {L} layers together.")
    fh = 230
    ctop = y
    rbox(ax, cx, ctop - lh, cw, lh, INK, radius=24, z=3)
    write(ax, cx + lp, ctop - lp - 10, "loss for each example", 24, True, "#A1A1A6", ha="left")
    yy = ctop - lp - 60
    for (t, sz, b, c), hl in zip(lines, hs):
        rich(ax, cx + lp, yy, t, sz, cw - 2 * lp, [(0, len(t), "bold")] if b else (), color=c)
        yy -= hl + 18
    arrow(ax, (16 + 2 * pw + gap + 12, ctop - lh / 2), (cx - 12, ctop - lh / 2), lw=5, head=30)
    # one backward pass reaches the LoRA of every layer
    sx, ex = cx + 40, cx + cw - 40
    step = (ex - sx) / L
    bw = step * 0.66
    btop = ctop - lh - 150
    comb = btop + 60
    ax.plot([cx + cw / 2] * 2, [ctop - lh - 4, comb], color=PURPLE, lw=4, zorder=4)
    write(ax, cx + cw / 2 + 20, ctop - lh - 46, "one backward pass", 24, True, PURPLE, ha="left")
    ax.plot([sx + bw / 2, ex - step + bw / 2], [comb] * 2, color=PURPLE, lw=3, zorder=4, solid_capstyle="round")
    for l in range(L):
        c = sx + l * step + bw / 2
        ax.add_patch(FancyArrowPatch((c, comb), (c, btop + 2), arrowstyle="-|>,head_length=0.45,head_width=0.3",
                                     color=PURPLE, lw=1.8, mutation_scale=20, zorder=4, shrinkA=0, shrinkB=0))
        rbox(ax, sx + l * step, btop - 150, bw, 150, PURPLE_LIGHT, radius=5, z=3, edge=PURPLE, lw=1.6)
    write(ax, sx, btop - 180, "layer 0", 22, color=MUTED, ha="left")
    write(ax, ex - step + bw, btop - 180, f"layer {L - 1}", 22, color=MUTED, ha="right")
    write(ax, cx + cw / 2, btop - 220, "a rank-32 LoRA on every attention and MLP matrix", 22, True, PURPLE)
    rich(ax, cx, btop - fh - 40 - 10, note, 26, cw, [(0, len(note), "bold")], color=PURPLE)
    y = yb - 60
    colour_scale(ax, 16 + pw - 200, y, w=340, size=24)
    rbox(ax, 16 + pw + 170, y - 16, 44, 32, "none", radius=6, z=5, edge=INK, lw=3, ls=(0, (4, 3)))
    write(ax, 16 + pw + 228, y, "target", 24, True, ha="left")
    crop(fig, ax, y - 80, 4000)
    save(fig, "training")


# 4. What training saw, and what it never saw.

LAYER_NAMES = {"all": "all layers", "range": "a range", "first": "first k", "last": "last k", "region": "a third",
               "half": "a half", "single": "one layer", "list": "a list", "parity": "even or odd",
               "every3": "every 3rd", "ramp_up": "ramp up", "ramp_down": "ramp down", "bump": "a bump",
               "every45": "every 4th or 5th", "prime": "prime", "square": "perfect squares", "power2": "powers of two",
               "except": "all but a range", "vshape": "V shape", "wave": "wave", "numset": "number sets",
               "complement": "all but a set", "union": "two sets", "list_long": "a long list",
               "every_m": "every 3rd,\n6th, or 7th", "ramp_part": "partial ramp", "valley": "a valley",
               "blocks": "blocks", "levels": "levels"}
TOKEN_NAMES = {"all": "every token", "first": "first n tokens", "after": "after n tokens", "grow": "growing",
               "alternate": "alternating", "fade": "fading", "between": "a token span", "window": "a window",
               "every_k": "every 3rd\nor 4th"}


def strip(ax, x, y, w, h, f):
    n = len(f)
    cw = w / n
    for i, v in enumerate(f):
        ax.add_patch(Rectangle((x + i * cw, y), cw * 0.82, h, fc=DIV((np.clip(2 * v, -LIM, LIM) + LIM) / (2 * LIM)),
                               ec="none", zorder=3))


def split():
    """The axes of an instruction, as the r4 run was trained and tested (spec.py)."""
    fig, ax = canvas(4000)
    y = 3960
    lx, ax0, bx0, cw = 30, 250, 735, 440
    hs = fit(ax, "Never seen in training", 34, cw - 30, True)
    write(ax, ax0 + cw / 2, y - 20, "In training", hs, True)
    pill(ax, bx0 + cw / 2 - (text_width(ax, "Never seen in training", hs, True) + 37) / 2, y - 20,
         "Never seen in training", hs, NEW)
    y -= 70
    top = y

    def row(label, h):
        nonlocal y
        write(ax, lx, y - 14, label, 28, True, ha="left", va="top", linespacing=1.25)
        y0 = y
        y -= h
        return y0

    def text_cells(y0, a, b, size=25):
        ya = rich(ax, ax0 + 20, y0 - 18, a, size, cw - 40)
        yb = rich(ax, bx0 + 20, y0 - 18, b, size, cw - 40)
        return min(ya, yb)

    entries = [("Concepts", "135, such as volcanoes, wizards, pirates, thunder, and the moon",
                "45 others, such as robots, stars, snow, whales, and chess"),
               ("Wording", "20 verbs (raise, boost, turn down …), 7 sentence forms, 12 openings",
                "8 other verbs (crank up, heighten, quiet down …), 3 sentence forms, 5 openings")]
    for label, a, b in entries:
        y0 = row(label, 0)
        y = text_cells(y0, a, b) - 40
    for label, fams_a, fams_b, n, names, pattern in (
            ("Layers", spec.LAYER_TRAIN, spec.LAYER_TEST, 36, LAYER_NAMES, spec.layer_pattern),
            ("Reply\ntokens", spec.TOKEN_TRAIN, spec.TOKEN_TEST, 64, TOKEN_NAMES, None)):
        y0 = row(label, 0)
        bottoms = []
        for col_x, fams in ((ax0, fams_a), (bx0, fams_b)):
            yy, extra = y0 - 20, 0
            per_row = 1 if len(fams) * 2 <= len(fams_a) + 1 else 2
            for k, fam in enumerate(fams):
                cx = col_x + 20 + (k % per_row) * (cw - 20) / 2
                if k % per_row == 0 and k:
                    yy -= 92 + extra
                    extra = 0
                rng = random.Random(3)
                if pattern:
                    f, _ = pattern(rng, fam, n)
                else:
                    f, _ = spec.token_pattern(rng, fam, n)
                sw = 170 if per_row == 1 else (cw - 60) / 2
                strip(ax, cx, yy - 26, sw, 26, f)
                if per_row == 1:
                    write(ax, cx + sw + 14, yy - 13, names[fam], 23, ha="left")
                elif "\n" in names[fam]:
                    write(ax, cx, yy - 40, names[fam], 23, ha="left", va="top", linespacing=1.1)
                    extra = 34
                else:
                    write(ax, cx, yy - 56, names[fam], 23, ha="left")
            bottoms.append(yy - 92 - extra)
        y = min(bottoms) - 10
    for label, a, b in (("Edits", "1 or 2 concepts per instruction", "3 concepts in one instruction"),
                        ("Questions", "232 questions", "78 other questions, in every test")):
        y0 = row(label, 0)
        y = text_cells(y0, a, b) - 40
    rbox(ax, bx0 - 10, y + 10, cw + 20, top - y + 4, NEW_PANEL, radius=22, z=1)
    rbox(ax, ax0 - 10, y + 10, cw + 20, top - y + 4, PANEL, radius=22, z=1)
    crop(fig, ax, y - 20, 4000)
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


def scores():
    """Each test set: the share of the requested change and the pattern match r of the trained model and the original
    model (mean and 95% bootstrap interval over 96 instructions; results/eval_r4.json)."""
    E, _ = data()
    fig, ax = canvas(3000)
    y = 2960
    lab_w = 390
    pct = lambda v: "0%" if round(100 * v) == 0 else f"{100 * v:.0f}%".replace("-", "−")
    dec = lambda v: "0.00" if round(v, 2) == 0 else f"{v:.2f}".replace("-", "−")
    panels = (("share of the", "requested change", 420, 760, 0.0, 1.0, ((0, "0%"), (0.5, "50%"), (1.0, "100%")),
               pct, "frac"),
              ("pattern", "match r", 800, 1100, -0.4, 0.8, ((0, "0"), (0.4, "0.4"), (0.8, "0.8")), dec, "r"))
    for h1, h2, a, b, lo, hi, *_ in panels:
        z = a + (b - a) * (0 - lo) / (hi - lo)
        write(ax, z, y - 20, h1, 26, True, ha="left")
        write(ax, z, y - 56, h2, 26, True, ha="left")
    y -= 110
    top = y + 20
    for key, label in SCORE_ROWS:
        if key == "seen":
            pill(ax, 24, y, "In training", 24, PANEL)
            y -= 56
        if key == "new_wording":
            y -= 24
            pill(ax, 24, y, "Never seen in training", 24, NEW)
            y -= 56
        S = E["sets"][key]["summary"]
        rows_bottom = []
        for _, _, a, b, lo, hi, ticks, fmt, k in panels:
            m, cl, ch = S[k][:3]
            o = S["orig"][k][0]
            rows_bottom.append(hbars(ax, a, b, lo, hi, y, [m, o], [PURPLE, ORIG], [34, 14], fmt,
                                     cis=[(cl, ch), None]))
        lb = rich(ax, 30, y + 6, label, 27, lab_w - 20)
        y = min(min(rows_bottom), lb) - 22
    for _, _, a, b, lo, hi, ticks, fmt, k in panels:
        axis(ax, a, b, lo, hi, y, ticks, top)
    y -= 70
    x = 30
    for lab, col in (("trained model", PURPLE), ("original Qwen3-4B, thinking off", ORIG)):
        rbox(ax, x, y - 15, 30, 30, col, radius=6)
        write(ax, x + 42, y, lab, 25, True, ha="left")
        x += 42 + text_width(ax, lab, 25, True) + 50
    x, y = 30, y - 56
    ax.plot([x, x + 40], [y, y], color=INK, lw=2)
    for e in (x, x + 40):
        ax.plot([e] * 2, [y - 7, y + 7], color=INK, lw=2)
    write(ax, x + 52, y, "95% interval", 25, True, ha="left")
    crop(fig, ax, y - 40, 3000)
    save(fig, "scores")


# 5b. What the two scores are computed from: every cell of one instruction.

CONCEPT_COLOURS = ("#1A9E77", "#D95F02", "#2A6FDB")
SET_TAGS = {"seen": ("Only this question is new", ()),
            "new_wording": ("Training never saw this wording", ("wording",)),
            "new_concepts": ("Training never saw this concept", ("concept",)),
            "three_edits": ("Training never saw three edits at once", ("three",)),
            "new_layer_patterns": ("Training never saw this layer pattern", ("layers",))}
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

def heat_rows(ax, y, rows, columns, hw, gap=20, size=27, ox=0):
    """One panel per row: a tag, the instruction (parts never seen in training highlighted) and one heatmap per column
    and requested concept, with the trained model's plan in a thinking block, in a column of width W starting at ox.
    rows: (tag, new, item, profile); columns: (header, profile key). Returns the bottom edge."""
    x0 = ox + (W - len(columns) * hw - (len(columns) - 1) * gap) / 2 + 30
    hh = hw * 36 / 64
    for k, (head, _) in enumerate(columns):
        write(ax, x0 + k * (hw + gap) + hw / 2, y, head, 30, True, PURPLE if head == "trained model" else INK)
    y -= 44
    for tag, new, it, prof in rows:
        text = instruction_only(it["user"], TASKS[it["task"]])
        marks = marks_for(text, it, new)
        nc = len(it["concepts"])
        sub = 40 if nc > 1 else 0
        th = rich_height(text, size, W - 100, marks)
        ph = 24 + 40 + plan_height(it["plan"], 20, W - 120) + 44 if it.get("plan") else 0
        h = 30 + 50 + th + ph + 30 + nc * (hh + sub) + (nc - 1) * 18 + 30
        rbox(ax, ox + 16, y - h, W - 32, h, PANEL, radius=26, z=1)
        pill(ax, ox + 40, y - 44, tag, 22, NEW if new else "white", bold=True)
        ty = rich(ax, ox + 40, y - 80, text, size, W - 100, marks)
        if ph:
            ty = thinking_block(ax, ox + 40, ty - 24, W - 80, it["plan"], False, size=20,
                                label="the trained model's thinking block")
        hy = ty - 30
        for c in range(nc):
            top = hy - c * (hh + sub + 18) - sub
            if nc > 1:
                write(ax, x0, top + 22, K.NAMES[it["concepts"][c]], 25, True, ha="left")
            for k, (_, key) in enumerate(columns):
                heat(ax, x0 + k * (hw + gap), top - hh, hw, hh, grid(prof, key, c))
            heat_axes(ax, x0, top - hh, hw, hh, prof["target"].shape[1], size=20, tokens=False)
        y -= h + 18
    for k in range(len(columns)):
        xx = x0 + k * (hw + gap)
        write(ax, xx + 4, y - 4, "0", 20, color=MUTED, ha="left")
        write(ax, xx + hw / 2, y - 4, "reply token", 20, color=MUTED)
        write(ax, xx + hw - 4, y - 4, str(T_SHOWN - 1), 20, color=MUTED, ha="right")
    return y - 40


def examples():
    """For four test sets, the instruction closest to the median pattern match of the set: requested, trained model
    and original model, per layer and reply token."""
    E, P = data()
    rows = []
    for s in ("seen", "new_wording", "new_concepts", "three_edits"):
        tag, new = SET_TAGS[s]
        i = typical(E["sets"][s]["items"])
        rows.append((tag, new, E["sets"][s]["items"][i], P[s][i]))
    fig, ax = canvas(4000, 2 * W)
    columns = (("requested", "target"), ("trained model", "dz"), ("original model", "orig_dz"))
    y = min(heat_rows(ax, 3960, rows[:2], columns, 350), heat_rows(ax, 3960, rows[2:], columns, 350, ox=W))
    colour_scale(ax, W, y - 10)
    crop(fig, ax, y - 80, 4000)
    save(fig, "examples")


def outside():
    """Layer and reply-token patterns never seen in training: for each rule, the instruction closest to the median
    pattern match of that rule; side by side."""
    E, P = data()
    rows = []
    for s, fam, tag, new in (("new_layer_patterns", "prime", "Training never saw prime-numbered layers",
                              ("layers",)),
                             ("new_layer_patterns", "power2", "Training never saw powers of two", ("layers",)),
                             ("new_token_patterns", "tok_alternate", "Training never saw alternating tokens",
                              ("tokens",)),
                             ("new_token_patterns", "tok_fade", "Training never saw fading across reply tokens",
                              ("tokens",))):
        items = E["sets"][s]["items"]
        i = typical(items, lambda it: fam in it["fams"])
        rows.append((tag, new, items[i], P[s][i]))
    fig, ax = canvas(5000, 2 * W)
    columns = (("requested", "target"), ("trained model", "dz"))
    y = min(heat_rows(ax, 4960, rows[:2], columns, 500), heat_rows(ax, 4960, rows[2:], columns, 500, ox=W))
    colour_scale(ax, W, y - 10)
    crop(fig, ax, y - 80, 5000)
    save(fig, "outside")


# A plan in a thinking block.

def mono_width(ax, size):
    return text_width(ax, "0" * 20, size, family=MONO) / 20


def plan_text(ax, x, top, text, size, width, wrong):
    """The plan as written, in a monospaced font wrapped to width; layer numbers in `wrong` are red and bold.
    Returns the bottom edge."""
    cw = mono_width(ax, size)
    per = max(int(width // cw), 10)
    step = size * PT * 1.45
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
    return y + step - size * PT * 0.7


def thinking_block(ax, x, top, w, text, prefilled, size=22, label="thinking block"):
    """A plan inside a thinking block whose top-left corner is (x, top): lavender with a purple dashed edge when the
    trained model generates the plan, white with a grey dashed edge when the plan is prefilled, and empty when text is
    None (the chat template's empty block with thinking turned off). Returns the bottom."""
    write(ax, x, top - 14, label, 22, True, MUTED, ha="left")
    if prefilled or text is None:
        pill(ax, x + text_width(ax, label, 22, True) + 16, top - 14, "prefilled" if prefilled else "thinking off", 20,
             REPLY, MUTED)
    top -= 40
    if text is None:
        rbox(ax, x, top - 56, w, 56, "white", radius=20, z=3, edge=MUTED, lw=2.5, ls=(0, (5, 4)))
        return top - 56
    h = plan_height(text, size, w - 40) + 44
    rbox(ax, x, top - h, w, h, "white" if prefilled else LAVENDER, radius=20, z=3, edge=MUTED if prefilled else PURPLE,
         lw=2.5, ls=(0, (5, 4)))
    plan_text(ax, x + 20, top - 22, text, size, w - 40, set())
    return top - h


def plan_height(text, size, width):
    fig, ax = canvas(2000)
    h = 2000 - plan_text(ax, 0, 2000, text, size, width, set())
    plt.close(fig)
    return h


FIGURES = ("overview", "recipe", "training", "split", "scores", "datapoints", "examples", "outside")

if __name__ == "__main__":
    style()
    TASKS = json.loads((ROOT / "magician" / "tasks.json").read_text())
    for n in sys.argv[1:] or FIGURES:
        globals()[n]()
