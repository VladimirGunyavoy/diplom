"""b3: какие флаги/умолчания у ДИ (solve8.ENV + PLUS) и у pend (pend_env.sh + мои) — сравнение эффективных значений. python flag_diff.py di|pend"""
import os, sys, json
sys.path.insert(0, os.path.abspath('.')); sys.path.insert(0, os.path.join(os.path.abspath('.'), 'src', 'algo'))
which = sys.argv[1]
if which == 'di': os.environ.update(SOLVE8_PLUS='1')
from src.algo import solve8 as S
g = S.load_growN(); out = {}
for k in 'SYS M DTN RMAX TMAX DELTA KF OVH FRAC RMIN MINROWS GNEAR NFAIL QB SEEDNEAR GOALFRONT GOALW GOALB GLIM GSEED GOALSHAPE NORMFRONT FMIN ZACEP ZDEPTH COVTOL COVN COVP GM RMINZ ROWSMIN SIDE QMIX SMAX XLV RHO'.split():
    v = getattr(g, k, None); out[k] = v.tolist() if hasattr(v, 'tolist') else v
G = S.make_graph(g, {u: [] for u in g.US}, 'N') if False else None
print(json.dumps(out, default=str))
