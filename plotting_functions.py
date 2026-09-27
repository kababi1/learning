import numpy as np
import matplotlib.pyplot as plt


L_VAPOR      = 2.5e-3   # length of the vapor grid [m]
PLOT_RHO_MIN = 1e-5     # cells below this density [kg/m^3] count as vacuum and are not drawn
M_CU         = 1.055e-25  # copper atom mass [kg]

# colorblind-checked categorical colors, used in this order
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']


def x_mm_axis(nx):
    """Cell-centre positions of the vapor grid in mm."""
    return (np.arange(nx) + 0.5) * (L_VAPOR / nx) * 1e3


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color='0.9', lw=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def plume_mask(rho_i, rho_min=PLOT_RHO_MIN):
    """True on the plume: the longest contiguous run of cells with rho >= rho_min."""
    mk = rho_i >= rho_min
    idx = np.flatnonzero(mk)
    if idx.size == 0:
        return mk
    runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    best = max(runs, key=len)
    mk = np.zeros_like(mk)
    mk[best] = True
    return mk


def masked_profile(field_i, rho_i, rho_min=PLOT_RHO_MIN):
    """Copy of field_i with NaN outside the plume, so matplotlib leaves a gap there instead of drawing zeros."""
    out = np.full(field_i.shape, np.nan)
    mk = plume_mask(rho_i, rho_min)
    out[mk] = field_i[mk]
    return out


def frames_to_plot(n_frames, every=4):
    """Every `every`-th frame plus the last one."""
    idx = list(range(0, n_frames, every))
    if idx[-1] != n_frames - 1:
        idx.append(n_frames - 1)
    return idx


def plot_profiles(ax, field, rho, times, ylabel, every=4, rho_min=PLOT_RHO_MIN, logy=False):
    """Plot spatial profiles of `field` (shape [n_frames, nx]) at several times, plume only."""
    x_mm = x_mm_axis(field.shape[1])

    frames = frames_to_plot(len(times), every)
    # times are ordered, so use one hue from light (early) to dark (late)
    colors = plt.cm.Blues(np.linspace(0.35, 1.0, len(frames)))

    x_front = 0.0
    for color, i in zip(colors, frames):
        y = masked_profile(field[i], rho[i], rho_min)
        if np.all(np.isnan(y)):
            continue   # no plume yet (e.g. t = 0)
        ax.plot(x_mm, y, color=color, lw=1.5, label=f'{times[i] * 1e9:.0f} ns')
        x_front = max(x_front, x_mm[np.flatnonzero(~np.isnan(y))[-1]])

    if x_front > 0:
        ax.set_xlim(0, 1.05 * x_front)   # zoom to the plume, as in the paper
    if logy:
        ax.set_yscale('log')

    style_axes(ax, 'Distance from target (mm)', ylabel)
    ax.legend(title='Time', frameon=False, fontsize=9)


if __name__ == '__main__':
    with np.load('ablation_results.npz') as data:
        time_steps = data['time']
        T_vap      = data['T_vap']
        rho_vap    = data['rho_vap']

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_profiles(ax, T_vap, rho_vap, time_steps, 'Temperature (K)')
    fig.tight_layout()

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_profiles(ax, rho_vap, rho_vap, time_steps, r'Density (kg m$^{-3}$)', logy=True)
    fig.tight_layout()

    plt.show()
