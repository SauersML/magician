"""figures/gnome.png from results/gnome_r4.pt (magician/gnome.py), drawn with make_figures.py's primitives: gnomes, a
concept never seen in training, raised in the shape of a gnome hat over reply tokens and layers. Panels: the requested
change, and the mean change over 8 held-out questions of the trained model with its own plan and with the exact plan
prefilled, and of the original model with thinking off. Usage: python gnome_figure.py"""
import numpy as np
import torch

import make_figures as M


def gnome():
    G = torch.load(M.RES / "gnome_r4.pt", weights_only=False)
    L = G["hat"].shape[0]
    panels = (("Requested", G["hat"].T), ("Trained model", np.nanmean(G["own"]["dz"], 0)),
              ("Trained model,\nexact plan prefilled", np.nanmean(G["forced"]["dz"], 0)),
              ("Original model,\nthinking off", np.nanmean(G["orig"]["dz"], 0)))
    pw, ph, gap, x0, y0 = 420, 360, 80, 120, 190
    width = x0 + len(panels) * pw + (len(panels) - 1) * gap + 50
    top = y0 + ph + 250
    fig, ax = M.canvas(top, width)
    for i, (lab, A) in enumerate(panels):
        x = x0 + i * (pw + gap)
        M.heat(ax, x, y0, pw, ph, A)
        M.heat_axes(ax, x, y0, pw, ph, L, size=24, layers=i == 0)
        M.write(ax, x + pw / 2, y0 + ph + 70, lab, 32, True)
    lab = "Gnomes: concept never seen in training"
    M.pill(ax, width / 2 - (M.text_width(ax, lab, 28, True) + 28 * 1.1) / 2, top - 55, lab, 28, M.NEW)
    M.colour_scale(ax, width / 2, 70, size=28)
    M.save(fig, "gnome")


if __name__ == "__main__":
    M.style()
    gnome()
