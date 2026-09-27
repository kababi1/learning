import numpy as np
import matplotlib.pyplot as plt
from plotting_functions import SERIES_COLORS, style_axes


def plot_shielding(ax, data):
    """Incident laser pulse vs. the intensity that reaches the target, in W/cm^2 against ns."""
    t = data['time_I']
    # the save arrays are preallocated; unused slots at the end are all zeros (t = 0)
    n = np.flatnonzero(t)[-1] + 1
    t_ns = t[:n] * 1e9
    I_in = data['I_no_shielding'][:n] * 1e-4   # W/m^2 -> W/cm^2
    I_tg = data['I'][:n] * 1e-4

    ax.plot(t_ns, I_in, color=SERIES_COLORS[0], lw=1.8, label='Incident laser pulse')
    ax.plot(t_ns, I_tg, color=SERIES_COLORS[1], lw=1.8, ls='--', label='Reaching the target')
    ax.set_xlim(0, t_ns[-1])
    ax.set_ylim(0, 1.05 * I_in.max())
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    style_axes(ax, 'Time (ns)', r'Laser irradiance (W cm$^{-2}$)')
    ax.legend(frameon=False, fontsize=9)


if __name__ == '__main__':
    with np.load('ablation_results.npz') as data:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_shielding(ax, data)
        fig.tight_layout()
        plt.show()
