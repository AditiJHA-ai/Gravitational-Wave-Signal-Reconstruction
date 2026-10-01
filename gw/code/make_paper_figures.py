import os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C_LSTM, C_TF, C_REF = '#1f77b4', '#d95f02', '#6e6e6e'
FS = 1024
OUT = 'figures_paper'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 12, 'axes.labelsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'lines.linewidth': 2.0, 'lines.markersize': 7,
    'legend.frameon': False, 'figure.dpi': 110,
})

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote', path)

def load_pkl(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def fig_parameter_space(bank_file='bank_jitter.npz'):
    d = np.load(bank_file)
    keys = list(d['param_keys'])
    im1, im2 = keys.index('m1'), keys.index('m2')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    sets = [('train', 'Train', C_LSTM, 0.25, 6),
            ('ood_q', 'OOD mass ratio', C_TF, 0.7, 10),
            ('ood_spin', 'OOD spin', '#117733', 0.7, 10),
            ('ood_mass', 'OOD total mass', '#882255', 0.7, 10)]
    for k, lab, c, a, sz in sets:
        p = d[f'params_{k}']
        m1, m2 = p[:, im1], p[:, im2]
        axes[0].scatter(m1 + m2, m1 / m2, s=sz, c=c, alpha=a, label=lab,
                        edgecolors='none')
    axes[0].set_xlabel(r'Total mass $M_{\rm tot}$ ($M_\odot$)')
    axes[0].set_ylabel(r'Mass ratio $q = m_1/m_2$')
    axes[0].set_title('Parameter coverage')
    axes[0].legend(loc='upper right', fontsize=9)

    isz = keys.index('s1z')
    for k, lab, c, a, sz in sets:
        p = d[f'params_{k}']
        axes[1].scatter(p[:, im1] + p[:, im2], p[:, isz], s=sz, c=c, alpha=a,
                        edgecolors='none')
    axes[1].set_xlabel(r'Total mass $M_{\rm tot}$ ($M_\odot$)')
    axes[1].set_ylabel(r'Aligned spin $\chi_1$')
    axes[1].set_title('Spin coverage')
    fig.tight_layout()
    save(fig, 'fig_parameter_space.png')


def fig_training_curves(final='multiseed_results_final.pkl'):
    r = load_pkl(final)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, c, lab in [('lstm', C_LSTM, 'LSTM'), ('transformer', C_TF, 'Transformer')]:
        for metric, ax, ylab in [('val_loss', axes[0], 'Validation loss'),
                                 ('val_overlap_tf', axes[1], 'Validation overlap')]:
            series = []
            for rec in r[key]:
                h = rec['history']
                k = metric if metric in h else ('val_overlap_metric'
                                                if 'val_overlap_metric' in h else None)
                if k is None:
                    continue
                series.append(np.asarray(h[k], dtype=float))
            if not series:
                continue
            n = min(len(s) for s in series)
            M = np.stack([s[:n] for s in series])
            ep = np.arange(1, n + 1)
            ax.plot(ep, M.mean(0), color=c, label=lab)
            ax.fill_between(ep, M.min(0), M.max(0), color=c, alpha=0.18, linewidth=0)
            ax.set_xlabel('Epoch'); ax.set_ylabel(ylab)
    axes[0].set_yscale('log')
    axes[0].set_title('Training convergence')
    axes[1].set_title('Validation overlap')
    axes[1].axhline(0.3638, color=C_REF, ls='--', lw=1.5, label='AR-64 baseline')
    axes[0].legend(); axes[1].legend(loc='lower right')
    fig.tight_layout()
    save(fig, 'fig_training_curves.png')


def fig_seed_distribution():
    protos = [('multiseed_results.pkl', 'No warmup\npatience 7'),
              ('multiseed_results_warmup.pkl', 'Patience 15\n(no warmup)'),
              ('multiseed_results_final.pkl', 'Warmup 10\npatience 15')]
    have = [(f, lab) for f, lab in protos if os.path.exists(f)]
    if not have:
        print('skip seed distribution: no pkl files'); return
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    width = 0.16
    for i, (f, lab) in enumerate(have):
        r = load_pkl(f)
        for j, (key, c) in enumerate([('lstm', C_LSTM), ('transformer', C_TF)]):
            vals = [rec['val_overlap'] for rec in r.get(key, [])]
            if not vals:
                continue
            x = np.full(len(vals), i + (j - 0.5) * 0.32)
            x = x + np.linspace(-width / 2, width / 2, len(vals))
            ax.scatter(x, vals, s=55, c=c, alpha=0.9, edgecolors='white',
                       linewidths=0.8, zorder=3,
                       label=('LSTM' if j == 0 else 'Transformer') if i == 0 else None)
            ax.plot([i + (j - 0.5) * 0.32 - 0.11, i + (j - 0.5) * 0.32 + 0.11],
                    [np.mean(vals)] * 2, color=c, lw=2.5, zorder=2)
    ax.axhline(0.3638, color=C_REF, ls='--', lw=1.5, label='AR-64 baseline')
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([lab for _, lab in have])
    ax.set_ylabel('Validation overlap')
    ax.set_title('Per-seed outcomes under three training protocols')
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    save(fig, 'fig_seed_distribution.png')


def _binned(x, y, nbins=10):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    ctr, mu, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() < 5:
            continue
        ctr.append(0.5 * (a + b)); mu.append(np.mean(y[m]))
        lo.append(np.percentile(y[m], 25)); hi.append(np.percentile(y[m], 75))
    return np.array(ctr), np.array(mu), np.array(lo), np.array(hi)


def fig_overlap_vs(arrays='arrays_for_figures.npz', bank_file='bank_jitter.npz'):
    a = np.load(arrays)
    need = ['ov_lstm', 'ov_trans', 'snr_val']
    if not all(k in a.files for k in need):
        print('skip overlap-vs: missing arrays'); return
    ovl, ovt, snr = a['ov_lstm'], a['ov_trans'], a['snr_val']
    n = min(len(ovl), len(ovt), len(snr))
    ovl, ovt, snr = ovl[:n], ovt[:n], snr[:n]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for y, c, lab in [(ovl, C_LSTM, 'LSTM'), (ovt, C_TF, 'Transformer')]:
        cx, mu, lo, hi = _binned(snr, y)
        axes[0].plot(cx, mu, 'o-', color=c, label=lab)
        axes[0].fill_between(cx, lo, hi, color=c, alpha=0.16, linewidth=0)
    axes[0].axhline(0.3638, color=C_REF, ls='--', lw=1.5, label='AR-64 baseline')
    axes[0].set_xlabel('Injected network SNR'); axes[0].set_ylabel('Overlap')
    axes[0].set_title('Fidelity vs signal strength')
    axes[0].legend(loc='lower right', fontsize=9)

    pv = a['params_val'] if 'params_val' in a.files else None
    if pv is not None:
        d = np.load(bank_file); keys = list(d['param_keys'])
        mt = pv[:n, keys.index('m1')] + pv[:n, keys.index('m2')]
        for y, c, lab in [(ovl, C_LSTM, 'LSTM'), (ovt, C_TF, 'Transformer')]:
            cx, mu, lo, hi = _binned(mt, y)
            axes[1].plot(cx, mu, 'o-', color=c, label=lab)
            axes[1].fill_between(cx, lo, hi, color=c, alpha=0.16, linewidth=0)
        axes[1].axhline(0.3638, color=C_REF, ls='--', lw=1.5)
        axes[1].set_xlabel(r'Total mass $M_{\rm tot}$ ($M_\odot$)')
        axes[1].set_ylabel('Overlap')
        axes[1].set_title('Fidelity vs total mass (in distribution)')
        axes[1].legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    save(fig, 'fig_overlap_vs.png')


if __name__ == '__main__':
    for fn in [fig_parameter_space, fig_training_curves,
               fig_seed_distribution, fig_overlap_vs]:
        try:
            fn()
        except Exception as e:
            print(f'SKIP {fn.__name__}: {type(e).__name__}: {e}')
    print('\nDone. Figures in', OUT)
