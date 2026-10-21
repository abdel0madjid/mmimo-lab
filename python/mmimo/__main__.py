import argparse
from .cell import SCENARIOS, advise

p = argparse.ArgumentParser(description="Rank beam strategies for a massive-MIMO scenario")
p.add_argument("--scenario", choices=SCENARIOS, default="stadium")
p.add_argument("--top", type=int, default=5)
a = p.parse_args()
for score, c, r in advise(SCENARIOS[a.scenario])[:a.top]:
    print(f"score {score:.3f} | set #{c.sset} MU-BF={c.mu_bf} refine={c.refine} | cell {r['cell_mbps']:.0f} Mbps "
          f"edge {r['edge_mbps']:.0f} Mbps layers {r['layers']:.1f} SINR {r['avg_sinr_db']:.1f} dB")
