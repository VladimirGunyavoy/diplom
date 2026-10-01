import sys, os, time; sys.path.insert(0, '.')
os.environ['TUBES'] = '1'
src = open('tests/run_jet2.py').read().split("for st in")[0]
nov = sys.argv[1]; eps = float(sys.argv[2]); sys.argv = ['x', 'di', '14', nov]; exec(src)
from src.cells7.value import Field
t0 = time.time(); F = Field(S, layers, np.zeros(2), eps=eps); print('field', round(time.time() - t0), 'goal nodes', F.goal.sum(), flush=True); F.strict = True; F.solve(); print('sweeps', F.sweeps, 'finite', (F.V < 1e8).mean(), round(time.time() - t0), flush=True)
q = np.array([F.query(y) for y in Y]); r = q / ref; f = r[np.isfinite(r) & (q < 1e8)]
print('query finite', len(f) / len(r), 'median', np.median(f), 'mean', f.mean(), 'q10/q90', np.quantile(f, [.1, .9]))
