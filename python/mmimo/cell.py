"""Cell-level scenario: SSB beam sets, beam selection with hysteresis, SINR-driven TM adaptation, MU-BF pairing, strategy advisor."""
from dataclasses import dataclass, replace
import numpy as np
from .core import beam_amp, pathloss_umi, noise_dbm, channel_eigs, tm_rate, D2R

SETS = {"1": [1], "2": [2], "4": [4], "6": [6], "8": [8], "4#4": [4, 4], "5#3": [5, 3],
        "6#2": [6, 2], "3#3#2": [3, 3, 2], "2#2#2#2": [2, 2, 2, 2]}  # beams per elevation row


def ssb_beams(rows, az_span=120.0, el_span=45.0):
    """Beam centres: each row spans the sector in azimuth; rows are spread in elevation. Fewer beams -> wider (sub-array)."""
    R, out = len(rows), []
    for r, n in enumerate(rows):
        for k in range(n):
            out.append(dict(az=round(-az_span / 2 + az_span * (k + .5) / n, 1),
                            el=0.0 if R == 1 else round(el_span / 2 - el_span * (r + .5) / R, 1),
                            nx=max(1, int(0.85 * n + .5)), ny=max(1, int(2.27 * R + .5))))
    return out


@dataclass(frozen=True)
class Config:
    nx: int = 8; ny: int = 8; d: float = 0.5
    sset: str = "4#4"; fr2: bool = True; los: bool = False
    pw_dbm: float = 34.0; rho: float = 0.3; M: int = 4
    n_ue: int = 8; speed: float = 8.0; dist: str = "uniform"; range_m: float = 100.0
    mu_bf: bool = True; refine: bool = False; th_db: float = 10.0


SCENARIOS = {
    "cell": Config(),
    "stadium": Config(nx=16, ny=8, los=True, pw_dbm=37, n_ue=32, speed=0, dist="stadium", range_m=200),
    "concert": Config(nx=16, ny=8, sset="6#2", pw_dbm=37, n_ue=36, speed=2, dist="concert", range_m=250),
    "exit": Config(n_ue=24, speed=25, range_m=150),
    "festival": Config(nx=8, ny=4, sset="8", fr2=False, pw_dbm=43, n_ue=24, speed=5, range_m=400),
}


def _spawn(kind, n, rng):
    if kind == "stadium": return rng.uniform(-15, 15, n)
    if kind == "concert": return rng.choice([-28, 28], n) + rng.uniform(-9, 9, n)
    return rng.uniform(-55, 55, n)


def simulate(cfg, ticks=40, seed=1):
    """Run the cell for `ticks` steps; returns averaged KPIs (identical crowd for identical seed)."""
    rng = np.random.default_rng(seed)
    beams = ssb_beams(SETS[cfg.sset])
    fc, bw = (28.0, 4e8) if cfg.fr2 else (3.5, 1e8)
    nz = noise_dbm(bw, cfg.fr2)
    lam = channel_eigs(cfg.M, cfg.rho, cfg.los, rng)
    n = cfg.n_ue
    az, el = _spawn(cfg.dist, n, rng), rng.uniform(-12, 4, n)
    dist, v = rng.uniform(.25, 1, n) * cfg.range_m, rng.choice([-1, 1], n)
    sb, switches = -np.ones(n, int), 0
    cap, sep = max(2, min(8, (cfg.nx * cfg.ny) >> 3)), max(5, 90 / cfg.nx)
    acc = dict(cell=[], edge=[], layers=[], paired=[], sinr=[]); share = {}
    for _ in range(ticks):
        az = az + v * cfg.speed / 40
        v = np.where(np.abs(az) > 58, -v, v)
        dist = np.clip(dist + rng.uniform(-3, 3, n), 15, cfg.range_m)
        g = np.stack([min(cfg.nx, b["nx"]) * min(cfg.ny, b["ny"]) *
                      beam_amp(min(cfg.nx, b["nx"]), min(cfg.ny, b["ny"]), cfg.d, (b["az"], b["el"]), (az, el)) ** 2
                      for b in beams], axis=1)
        best, idx = g.argmax(1), np.arange(n)
        if (sb < 0).any(): sb = best.copy()
        sw = g[idx, best] > 2 * g[idx, sb]  # 3 dB hysteresis
        switches += int(sw.sum()); sb = np.where(sw, best, sb)
        cs = np.maximum(0, np.cos(az * D2R) * np.cos(el * D2R))
        snr = 10 ** ((cfg.pw_dbm - pathloss_umi(dist, fc, cfg.los) - nz) / 10) * g[idx, sb] * cs * 3.16 * (2 if cfg.refine else 1)
        groups = []
        for u in np.argsort(-snr):
            tgt = next((G for G in groups if cfg.mu_bf and len(G) < cap and
                        all(np.hypot(az[w] - az[u], 1.5 * (el[w] - el[u])) > sep for w in G)), None)
            tgt.append(u) if tgt is not None else groups.append([u])
        thr, lay, sinr, paired = np.zeros(n), 0.0, np.zeros(n), 0
        for G in groups:
            for u in G:
                sn = snr[u] / (1 + 0.35 * (len(G) - 1)); sd = 10 * np.log10(max(sn, 1e-9)); sinr[u] = sd
                tm = 2 if sd < 0 else 3 if (cfg.speed >= 20 and sd > 5) else 7 if sd < cfg.th_db else 8
                se, layers = tm_rate(tm, sn / 1.6, lam)
                thr[u] = se * bw / 1e6 / len(groups) * (0.92 if cfg.refine else 1)
                lay += layers / len(groups); paired += len(G) > 1
                lbl = "MU-BF" if len(G) > 1 else f"TM{tm}"
                share[lbl] = share.get(lbl, 0) + 1 / len(groups)
        acc["cell"].append(thr.sum()); acc["edge"].append(np.percentile(thr, 10))
        acc["layers"].append(lay); acc["paired"].append(paired / n); acc["sinr"].append(sinr.mean())
    tot = sum(share.values())
    return dict(cell_mbps=np.mean(acc["cell"]), edge_mbps=np.mean(acc["edge"]), layers=np.mean(acc["layers"]),
                pairing=np.mean(acc["paired"]), avg_sinr_db=np.mean(acc["sinr"]), beam_changes=switches,
                prb_share={k: s / tot for k, s in share.items()})


def advise(cfg, ticks=15, seed=777):
    """Rank (SSB set x MU-BF x refinement) on the same crowd: score = 60 % cell + 40 % cell-edge throughput."""
    res = []
    for s in SETS:
        for mu in (False, True):
            for rf in (False, True):
                c = replace(cfg, sset=s, mu_bf=mu, refine=rf)
                res.append((c, simulate(c, ticks, seed)))
    mc, me = max(r["cell_mbps"] for _, r in res), max(r["edge_mbps"] for _, r in res) or 1
    return sorted(((0.6 * r["cell_mbps"] / mc + 0.4 * r["edge_mbps"] / me, c, r) for c, r in res), key=lambda x: -x[0])
