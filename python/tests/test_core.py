import numpy as np
from dataclasses import replace
from mmimo import beam_amp, channel_eigs, tm_rate, ssb_beams, SETS, simulate, Config, pathloss_umi


def test_peak_at_steering_direction():
    assert abs(beam_amp(8, 4, .5, (20, 5), (20, 5)) - 1) < 1e-9


def test_more_elements_narrower_beam():
    off = (30, 0)
    assert beam_amp(16, 4, .5, (20, 0), off) < beam_amp(4, 4, .5, (20, 0), off)


def test_correlation_reduces_second_eigenmode():
    rng = np.random.default_rng(0)
    low = np.mean([channel_eigs(4, .0, False, rng)[1] for _ in range(200)])
    high = np.mean([channel_eigs(4, .95, False, rng)[1] for _ in range(200)])
    assert high < low


def test_tm8_beats_tm7_at_high_snr():
    lam = channel_eigs(4, 0.1, False, np.random.default_rng(1))
    assert tm_rate(8, 1000, lam)[0] > tm_rate(7, 1000, lam)[0]


def test_beam_counts():
    assert len(ssb_beams(SETS["4#4"])) == 8 and len(ssb_beams(SETS["3#3#2"])) == 8


def test_nlos_loses_more_than_los():
    assert pathloss_umi(100, 28, False) > pathloss_umi(100, 28, True)


def test_mu_bf_helps_in_a_crowd():
    base = Config(nx=16, ny=8, n_ue=32, dist="stadium", los=True, range_m=200, speed=0)
    assert simulate(base, 10)["cell_mbps"] > simulate(replace(base, mu_bf=False), 10)["cell_mbps"]
