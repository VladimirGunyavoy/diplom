"""research-23: конвейер manip 4D (исправленная динамика) на равномерном посеве (GS 0, GLIM 0) с крупным DELTA — слои (CPU, Pool 4) → solve GPU
(SBCAUS, NOLATCH, STGPU/SOLVEGPU) → агент FINGRID 2 VF 1 на 20 стартах, эталон OCP research-23 (r23/manip_ref_ocp_20.npy). Снимок growN — wb5 09:32."""
import os, sys, time, pickle, subprocess, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import growN as G
import __main__; [setattr(__main__, k_, v_) for k_, v_ in vars(G).items() if isinstance(v_, type)]
TAG = os.environ['TAG']; LP = 'L_%s.pkl' % TAG; t0 = time.time()
if os.path.exists(LP): A = G.Atlas.__new__(G.Atlas); A.layers = pickle.load(open(LP, 'rb'))['layers']; A.finish(); tb = 0.
else:
    A = G.Atlas(); tb = time.time() - t0; pickle.dump(dict(layers=A.layers), open(LP, 'wb'))
print('[%s] слои' % TAG, [len(l) for l in A.layers], 'узлов', A.N, 'build %.0f с' % tb, flush=True)
if os.environ.get('BUILDONLY'): sys.exit()                                                                  # слои — системным python3 (venv-numpy на посеве ×10 медленнее), solve — venv (torch)
while True:                                                                                                  # GPU aida общая: ждать, пока занято < GPUMAX МБ
    used = int(subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout.split()[0])
    if used < int(os.environ.get('GPUMAX', 6000)): break
    time.sleep(30)
t1 = time.time(); A.solve(); ts = time.time() - t1
print('[%s] solve %.0f с, проходов %s, конечных %.3f' % (TAG, ts, A.n_it, float((A.V < G.BIG / 2).mean())), flush=True)
pickle.dump(dict(layers=A.layers, V=A.V), open('A_%s.pkl' % TAG, 'wb'))
R = np.load(os.path.expanduser('~/spore_v5/wb4/spores_2/v5chain/reports/research/r23/manip_ref_ocp_20.npy')); Q, ref = R[:, :4], R[:, 4]
t2 = time.time(); T, sw, _ = A.rollout(Q); tq = (time.time() - t2) / len(Q); r = T / ref; ok = np.isfinite(T)
print('[%s] дошли %d/20 | T/эт мед. %.3f mean %.3f max %.3f | %.0f мс/запрос | по стартам %s' % (TAG, ok.sum(), *(np.round([np.median(r[ok]), r[ok].mean(), r[ok].max()], 3) if ok.any() else [np.nan] * 3), 1000 * tq, np.round(r, 2).tolist()), flush=True)
