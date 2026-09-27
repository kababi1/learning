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


if __name__ == '__main__':
    with np.load('ablation_results.npz') as data:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_species(ax, data, frame=-1)
        fig.tight_layout()
        plt.show()
