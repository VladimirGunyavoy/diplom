import os, sys, cProfile, pstats
import numpy as np
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
from src.algo import solve8 as S, ref_di_disk as R
g = S.load_growN(); atlas, _ = S.build_atlas([1., -1., 0.], 0, par=False); G = S.make_graph(g, atlas, 'N'); G.set_calc('q'); G.solve(verbose=False)
X2 = S.starts(g, 200, 1); Tr = R.T_ref(X2, G.rho)
pr = cProfile.Profile(); pr.enable(); G.run_agent(X2, 3 * Tr + 2, **S.POLICIES['commit']); pr.disable(); pstats.Stats(pr).sort_stats('tottime').print_stats(12)
