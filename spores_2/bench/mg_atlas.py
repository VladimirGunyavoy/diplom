"""CA4-B2: атласы манипулятора с целью (0, q2g) для q2g = π/2, π (тот же growN, GC = центр цели при посеве GS; слои в абсолютных q). Вместе с базовым q2g=0 три атласа покрывают все цели
(зеркало (q,ω,τ) → −(q,ω,τ) + сдвиг q1; каждый ±.8 по q2g; b9 SPD1b).
  python3 mg_atlas.py build TAG Q2G      (CPU, system python3: слои; время — в заготовку) → L_TAG.pkl
  venv-sb3 mg_atlas.py edges TAG Q2G     (GPU, под flock: полный solve с SAVEE → кэш рёбер E_TAG_*.npy; время — в заготовку)
Рабочая папка ~/spore_v5/c1/manip_b9; результаты времени → data/mg_atlas_TAG.json."""
import os, sys, time, json, pickle
for k, v in dict(SYS='manip', KF='11', M='3', FRAC='.5', OVH='.5', MINROWS='3', GLIM='0', RMAX='.6', TMAX='3', SBCAUS='1', SBLAY='0', NOLATCH='1', FINGRID='2', VF='2', PESS='1', GPUMAX='1000000', GS='300', DELTA='.3', MAXC='3000', SONE='1', FINBOX='1', STGPU=os.environ.get('STGPU', '1' if sys.argv[1] == 'edges' else '0'), OMP_NUM_THREADS='2', PYTHONUNBUFFERED='1').items(): os.environ.setdefault(k, v)
WD = os.path.expanduser('~/spore_v5/c1/manip_b9'); sys.path.insert(0, WD); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, common
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
mode, tag, q2g = sys.argv[1], sys.argv[2], float(sys.argv[3]); G.GC = np.array([0., q2g, 0., 0.]); fn = os.path.join(common.DATA, 'mg_atlas_%s.json' % tag)
J = json.load(open(fn)) if os.path.exists(fn) else {}
if mode == 'build':
    t0 = time.time(); A = G.Atlas(); tb = time.time() - t0; pickle.dump(dict(layers=A.layers), open(os.path.join(WD, 'L_%s.pkl' % tag), 'wb')); J.update(q2g=q2g, t_layers=tb, cells=[len(l) for l in A.layers], nodes=int(A.N)); print('LAYERS', tag, '%.0f с' % tb, J['cells'], A.N, flush=True)
else:
    A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open(os.path.join(WD, 'L_%s.pkl' % tag), 'rb'))['layers']; A.finish(); os.environ['SAVEE'] = os.path.join(WD, 'E_%s' % tag); os.environ['SOLVEGPU'] = '1'
    t0 = time.time(); A.solve(); ts = time.time() - t0; J.update(t_solve_savee=ts, iters=int(A.n_it)); print('EDGES', tag, '%.0f с' % ts, flush=True)
json.dump(J, open(fn, 'w'), indent=1, default=float)
