import numpy as np, time, sys
from pycbc.waveform import get_td_waveform
from scipy.signal import resample_poly

FS_RAW = 4096
FS = 1024
CONTEXT_LEN = 1024
FORECAST_LEN = 512
SEG_LEN = CONTEXT_LEN + FORECAST_LEN
F_LOWER = 20.0

ALIGNED_OFFSET_S = 0.10
JITTER_OFFSET_S = (0.05, 0.45)

SPLITS = {
    'train':    dict(m1=(15., 60.),  q=(1., 4.), spin=(-0.6, 0.6)),
    'val':      dict(m1=(15., 60.),  q=(1., 4.), spin=(-0.6, 0.6)),
    'ood_q':    dict(m1=(15., 60.),  q=(4., 8.), spin=(-0.6, 0.6)),
    'ood_spin': dict(m1=(15., 60.),  q=(1., 4.), spin=(0.7, 0.95)),
    'ood_mass': dict(m1=(70., 110.), q=(1., 4.), spin=(-0.6, 0.6)),
}

SIZES = {'train': 12000, 'val': 2000, 'ood_q': 1500, 'ood_spin': 1500, 'ood_mass': 1500}
KEYS = ['m1', 'm2', 's1z', 's2z', 'inc', 'off']


def sample_params(rng, r, jitter):
    m1 = rng.uniform(*r['m1'])
    q = rng.uniform(*r['q'])
    m2 = max(m1 / q, 5.0)
    off = rng.uniform(*JITTER_OFFSET_S) if jitter else ALIGNED_OFFSET_S
    return dict(m1=float(m1), m2=float(m2),
                s1z=float(rng.uniform(*r['spin'])), s2z=float(rng.uniform(*r['spin'])),
                inc=float(rng.uniform(0., np.pi)), off=float(off))


def generate_waveform(p):
    try:
        hp, hc = get_td_waveform(approximant='IMRPhenomD',
                                 mass1=p['m1'], mass2=p['m2'],
                                 spin1z=p['s1z'], spin2z=p['s2z'],
                                 inclination=p['inc'],
                                 delta_t=1.0 / FS_RAW, f_lower=F_LOWER)
    except Exception:
        return None
    h = resample_poly(np.asarray(hp.data, dtype=np.float64), FS, FS_RAW)
    peak = int(np.argmax(np.abs(h)))
    end = peak + int(p['off'] * FS)
    start = end - SEG_LEN
    if start < 0 or end > len(h):
        return None
    seg = h[start:end]
    n = np.max(np.abs(seg))
    if not np.isfinite(n) or n < 1e-30:
        return None
    return (seg / n).astype(np.float32)


def build(n, r, seed, label, jitter):
    rng = np.random.RandomState(seed)
    W, P, tries, t0 = [], [], 0, time.time()
    while len(W) < n:
        tries += 1
        p = sample_params(rng, r, jitter)
        w = generate_waveform(p)
        if w is None:
            continue
        W.append(w)
        P.append([p[k] for k in KEYS])
    print(f'  {label}: {n} kept / {tries} tries, {time.time()-t0:.1f}s', flush=True)
    return np.stack(W), np.array(P, dtype=np.float32)


def main(mode, scale, outfile):
    jitter = (mode == 'jitter')
    out = {}
    for i, (k, r) in enumerate(SPLITS.items()):
        W, P = build(int(SIZES[k] * scale), r, 100 + i, k, jitter)
        out[f'bank_{k}'] = W
        out[f'params_{k}'] = P
    out['param_keys'] = np.array(KEYS)
    out['meta'] = np.array([FS, CONTEXT_LEN, FORECAST_LEN, int(jitter)])
    np.savez_compressed(outfile, **out)
    print('saved', outfile)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'jitter'
    scale = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    main(mode, scale, f'bank_{mode}.npz')
