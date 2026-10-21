"""Core radio models: UPA beam patterns, 3GPP TR 38.901 UMi path loss, MIMO channel eigenmodes, TM rate models."""
import numpy as np

D2R = np.pi / 180


def uv(az, el):
    """Direction cosines (u, v) for azimuth/elevation in degrees (array normal = x)."""
    az, el = np.asarray(az, float) * D2R, np.asarray(el, float) * D2R
    return np.sin(az) * np.cos(el), np.sin(el)


def af_lin(n, d, delta):
    """Normalised array factor |sum_m exp(j*2*pi*d*m*delta)| / n of an n-element uniform line (d in wavelengths)."""
    delta = np.asarray(delta, float)
    if n <= 1:
        return np.ones_like(delta)
    s = np.sin(np.pi * d * delta)
    safe = np.where(np.abs(s) < 1e-6, 1.0, s)
    return np.where(np.abs(s) < 1e-6, 1.0, np.abs(np.sin(n * np.pi * d * delta) / (n * safe)))


def beam_amp(nx, ny, d, steer, target):
    """Amplitude (peak = 1) of an nx x ny UPA steered to `steer`=(az, el), seen towards `target`=(az, el)."""
    u0, v0 = uv(*steer)
    u, v = uv(*target)
    return af_lin(nx, d, u - u0) * af_lin(ny, d, v - v0)


def pathloss_umi(dist_m, fc_ghz, los, h_diff=8.5):
    """3GPP TR 38.901 UMi street-canyon path loss (dB), 3D distance from horizontal distance and height difference."""
    d3 = np.hypot(dist_m, h_diff)
    pl_los = 32.4 + 21 * np.log10(d3) + 20 * np.log10(fc_ghz)
    if los:
        return pl_los
    return np.maximum(pl_los, 35.3 * np.log10(d3) + 22.4 + 21.3 * np.log10(fc_ghz))


def noise_dbm(bw_hz, fr2):
    """Thermal noise + noise figure + 3 dB implementation margin - UE receive gain (FR2)."""
    return -174 + 10 * np.log10(bw_hz) + (9 if fr2 else 7) + 3 - (5 if fr2 else 0)


def channel_eigs(M, rho, los, rng, k_db=9.0):
    """Eigenvalues (sigma^2 / M, descending) of one MxM Rician/Rayleigh realisation with exponential Kronecker correlation."""
    K = 10 ** (k_db / 10) if los else 0.0
    R = rho ** np.abs(np.subtract.outer(np.arange(M), np.arange(M)))
    L = np.linalg.cholesky(R + 1e-9 * np.eye(M))
    W = (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))) / np.sqrt(2)
    H_los = np.exp(1j * (rng.uniform(0, 2 * np.pi, (M, 1)) - rng.uniform(0, 2 * np.pi, (1, M))))
    H = np.sqrt(K / (K + 1)) * H_los + np.sqrt(1 / (K + 1)) * (L @ W @ L.T)
    return np.sort(np.linalg.svd(H, compute_uv=False) ** 2)[::-1] / M


def _cap(x):
    return min(7.4, np.log2(1 + x)) * 0.75  # 256QAM ceiling, 25 % overhead


def tm_rate(tm, snr, lam):
    """Spectral efficiency (bit/s/Hz) and layers for LTE/NR transmission mode `tm` (1-8) at linear `snr`."""
    lam = np.asarray(lam)
    M = len(lam)

    def sm(rmax, pen):
        best = (0.0, 1)
        for r in range(1, min(M, rmax) + 1):
            t = sum(_cap(snr * pen * lam[k] / r) for k in range(r))
            if t > best[0]:
                best = (t, r)
        return best

    if tm == 1: return _cap(snr), 1                     # single antenna
    if tm == 2: return _cap(snr * lam.sum()), 1         # Tx diversity
    if tm == 3: return sm(M, 0.6)                       # open-loop SM
    if tm == 4: return sm(M, 1.0)                       # closed-loop SM
    if tm in (5, 6, 7): return _cap(snr * lam[0]), 1    # MU / rank-1 precoding / beamforming
    if tm == 8: return sm(2, 1.0)                       # dual-layer beamforming
    raise ValueError(f"unsupported TM{tm}")
