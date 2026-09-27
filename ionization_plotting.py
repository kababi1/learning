import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import (M_CU, PLOT_RHO_MIN, SERIES_COLORS, masked_profile,
                                x_mm_axis, style_axes)


SPECIES = [  # (key in the .npz, label, line style, line width)
    ('x_0',     'Cu atoms',        '-',  1.8),
    ('x_i_1',   r'Cu$^+$ ions',    '--', 1.8),
    ('x_i_2',   r'Cu$^{2+}$ ions', '-.', 1.8),
    # n_e ~ n(Cu+) while Cu2+ is scarce, so draw electrons wider to stay visible on top
    ('x_e_vap', 'Electrons',       ':',  3.0),
]


def plot_species(ax, data, frame=-1, as_density=True, rho_min=PLOT_RHO_MIN, decades=8):
    """Atom, ion and electron profiles at one saved frame, plume only.

    as_density=True plots number densities n = x * rho / m [m^-3];
    False plots the fractions x (per heavy particle) as the code stores them.
    """
    rho = data['rho_vap'][frame]
    x_mm = x_mm_axis(rho.size)
    n_heavy = rho / M_CU

    y_max = 0.0
    for color, (key, label, ls, lw) in zip(SERIES_COLORS, SPECIES):
        y = data[key][frame] * (n_heavy if as_density else 1.0)
        y = masked_profile(y, rho, rho_min)
        ax.plot(x_mm, y, color=color, ls=ls, lw=lw, label=label)
        y_max = max(y_max, np.nanmax(y))

    ax.set_yscale('log')
    # the solver floors fractions at ~1e-10; keep those from stretching the axis
    ax.set_ylim(y_max * 10.0**-decades, y_max * 2)

    front = np.flatnonzero(~np.isnan(masked_profile(rho, rho, rho_min)))
    if front.size:
        ax.set_xlim(0, 1.05 * x_mm[front[-1]])

    ylabel = r'Density (m$^{-3}$)' if as_density else 'Fraction of heavy particles'
    style_axes(ax, 'Distance from target (mm)', ylabel)
    ax.set_title(f't = {data["time"][frame] * 1e9:.0f} ns', loc='left', fontsize=11)
    ax.legend(frameon=False, fontsize=9)


def nearest_frame(times, time_ns):
    """Index of the saved frame closest to time_ns."""
    return int(np.argmin(np.abs(times * 1e9 - time_ns)))


def label_curves(ax, x, curves, labels):
    """Put each label, with an arrow, where its curve is furthest from the others (Bogaerts style).

    `curves` are already scaled to the left axis (0..1) so distances are comparable.
    """
    ok = ~np.isnan(curves[0])
    idx = np.flatnonzero(ok)
    lo, hi = idx[0] + len(idx) // 10, idx[-1] - len(idx) // 10   # stay off the plume edges
    cand = np.arange(lo, hi)
    used = []
    for k, (y, label) in enumerate(zip(curves, labels)):
        others = np.delete(curves, k, axis=0)
        gap = np.min(np.abs(others[:, lo:hi] - y[lo:hi]), axis=0)
        for u in used:   # don't stack labels at the same x
            gap[np.abs(cand - u) < 0.2 * (hi - lo)] *= 0.2
        j = lo + int(np.nanargmax(gap))
        used.append(j)
        # side the nearest other curve is on: put the text on the opposite side
        below = np.nanmin(np.where(others[:, j] < y[j], y[j] - others[:, j], np.inf))
        above = np.nanmin(np.where(others[:, j] >= y[j], others[:, j] - y[j], np.inf))
        dy = 14 if below < above else -14
        if y[j] < 0.12: dy = 14      # keep the text inside the axes
        if y[j] > 0.88: dy = -14
        # text to the right of the arrow, except near the right axis
        dx, ha = (18, 'left') if j < lo + 0.6 * (hi - lo) else (-18, 'right')
        ax.annotate(label, xy=(x[j], y[j]), xytext=(dx, dy), textcoords='offset points',
                    ha=ha, va='center', fontsize=10,
                    arrowprops=dict(arrowstyle='->', lw=0.8, color='0.2', shrinkA=1, shrinkB=2))


def plot_fractions(ax, data, time_ns=15.0, rho_min=PLOT_RHO_MIN):
    """Fig. 5c of Bogaerts et al. (2003): x_Cu of Cu0, Cu+, Cu2+ on the left axis (0-1),
    x_e on the right axis (0-2), vs. distance from the target, at one time."""
    frame = nearest_frame(data['time'], time_ns)
    rho = data['rho_vap'][frame]
    x_mm = x_mm_axis(rho.size)

    x0  = masked_profile(data['x_0'][frame],     rho, rho_min)
    xi1 = masked_profile(data['x_i_1'][frame],   rho, rho_min)
    xi2 = masked_profile(data['x_i_2'][frame],   rho, rho_min)
    xe  = masked_profile(data['x_e_vap'][frame], rho, rho_min)

    ax.plot(x_mm, x0,  color='k', ls='--', lw=1.4)
    ax.plot(x_mm, xi1, color='k', ls='-',  lw=1.4)
    ax.plot(x_mm, xi2, color='k', ls='-',  lw=1.4)
    ax.set_ylim(0, 1)

    # x_e = x(Cu+) + 2 x(Cu2+) runs from 0 to 2, so it gets its own axis as in the paper
    ax_e = ax.twinx()
    ax_e.plot(x_mm, xe, color='k', ls='-.', lw=1.4)
    ax_e.set_ylim(0, 2)
    ax_e.set_ylabel(r'$x_e$')

    label_curves(ax, x_mm, np.array([x0, xi1, xi2, xe / 2]),
                 [r'Cu$^0$', r'Cu$^+$', r'Cu$^{2+}$', r'e$^-$ (right axis)'])

    idx = np.flatnonzero(~np.isnan(x0))
    if idx.size:
        ax.set_xlim(0, 1.05 * x_mm[idx[-1]])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel(r'$x_{Cu}$')
    for a in (ax, ax_e):
        a.spines['top'].set_visible(False)
    ax.set_title(f't = {data["time"][frame] * 1e9:.0f} ns', loc='left', fontsize=11)


if __name__ == '__main__':
    with np.load('ablation_results.npz') as data:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_fractions(ax, data, time_ns=15.0)
        fig.tight_layout()
        plt.show()
