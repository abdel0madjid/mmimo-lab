# mmimo-lab

Interactive 3D massive-MIMO / beamforming lab plus a Python reference model.
Plan a sector the way operators do: **SSB beam set -> SINR-driven transmission-mode (TM) adaptation -> MU-BF pairing -> KPIs**,
for event scenarios (stadium, concert, match exit, festival).

* `docs/index.html` - single-file 3D simulator (three.js). Open it in a browser, no build step. Served by GitHub Pages from `/docs`.
* `python/mmimo` - NumPy reference model of the same logic, with tests and a CLI strategy advisor.

## Quick start
```bash
pip install -e "python[dev]" && pytest -q python/tests
python -m mmimo --scenario stadium        # ranks 36 strategies (SSB set x MU-BF x beam refinement)
```
```python
from mmimo import SCENARIOS, simulate
print(simulate(SCENARIOS["concert"]))
```

## What is modelled
UPA beam patterns (separable array factor, element spacing), SSB beam sets (#1 ... #4#4, #5#3, #6#2, #3#3#2, #2#2#2#2),
beam selection with 3 dB hysteresis, TM1-TM8 rate models, Rayleigh/Rician MIMO channel with exponential correlation (rank from SVD),
3GPP TR 38.901 UMi path loss, greedy angular UE pairing for MU-BF, hybrid beamforming with quantised phase shifters (web app).

## Limitations (please read)
Single cell, no inter-cell interference, one shared channel realisation, simplified Shannon-gap throughput,
beam angles derived from sector geometry (3GPP defines SSB indices, not beam angles). Results compare strategies; they are **not** network predictions.
The Python port has not yet been cross-validated against external toolchains.

## Roadmap
Adapter for NVIDIA Sionna (TR 38.901 CDL/UMi channels), cross-checks vs MATLAB 5G Toolbox / QuaDRiGa, per-UE fading, inter-cell interference, export of results to the web viewer (JSON or Pyodide).

## License
MIT. No third-party course material or slides are included.
