# diff копий d4 к базовым файлам (только изменённые строки)

## c12/tab/tvg_vt.py -> d4/code/tvg_vt.py
< Образец — w38_car/car_val_w38.py. Запуск (aida, ~/spore_v5/tab/spores_2/w39): python tvg.py ENV NB K TREESEED
> Образец — w38_car/car_val_w38.py. Запуск (aida, ~/spore_v5/d4/spores_2/w39n): python tvg.py ENV NB K TREESEED
< HERE = os.path.expanduser('~/spore_v5/tab/spores_2/w39'); sys.path.insert(0, os.path.join(HERE, '..', 'bench')); sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
> HERE = os.path.expanduser('~/spore_v5/d4/spores_2/w39n'); sys.path.insert(0, os.path.join(HERE, '..', 'bench')); sys.path.insert(0, os.path.join(HERE, '..', 'v6'))
< def okv(P): P = np.asarray(P); return ~(((P < LO) | (P > HI)) & (env.period == 0)).any(-1)
> WMARG = float(os.environ.get("WMARG", 0)); _MG = [0.]   # d4: запас к коробке скоростей только в агенте (включается в main после VI: _MG[0] = WMARG)
> def okv(P): P = np.asarray(P); m = _MG[0] * (env.period == 0); return ~(((P < LO + m) | (P > HI - m)) & (env.period == 0)).any(-1)
<     json.dump(out, open(os.path.expanduser('~/spore_v5/tab/out/vt_%s_%d_%s_%s_%d.json' % (ENV, NB, os.environ.get('TAUS', str(TAU)), 'rnd' if os.environ.get('RNDN') == '1' else 'tree', TS)), 'w'), default=float)
>     json.dump(out, open(os.path.expanduser('~/spore_v5/d4/out/vt_%s_%d_%s_%s_%d.json' % (ENV, NB, os.environ.get('TAUS', str(TAU)), 'rnd' if os.environ.get('RNDN') == '1' else 'tree', TS)), 'w'), default=float)

## c12/tab/tvg_lp.py -> d4/code/tvg_lp.py
< sys.path.insert(0, os.path.expanduser('~/spore_v5/tab'))
> sys.path.insert(0, os.path.expanduser('~/spore_v5/d4/spores_2/w39n'))
>     T._MG[0] = T.WMARG
<     json.dump(out, open(os.path.expanduser('~/spore_v5/tab/out/lp_%s_%d_%g_%s_%d' + os.environ.get('TAG', '') + '.json') % (T.ENV, T.NB, T.TAU, 'rnd' if os.environ.get('RNDN') == '1' else 'tree', ts), 'w'), default=float
>     json.dump(out, open(os.path.expanduser('~/spore_v5/d4/out/lp_%s_%d_%g_%s_%d' + os.environ.get('TAG', '') + '.json') % (T.ENV, T.NB, T.TAU, 'rnd' if os.environ.get('RNDN') == '1' else 'tree', ts), 'w'), default=float)

## c12/tab/tvg_dl.py -> d4/code/tvg_dl.py
< sys.path.insert(0, os.path.expanduser('~/spore_v5/tab'))
> sys.path.insert(0, os.path.expanduser('~/spore_v5/d4/spores_2/w39n'))
<     out['diag'] = diag; json.dump(out, open(os.path.expanduser(('~/spore_v5/tab/out/dl_%s_%d_%s_%d' + ('_s100' if os.environ.get('S100')=='1' else '') + '.json') % (T.ENV, T.NB, os.environ.get('TAUS', str(T.TAU)), ts)), 
>     out['diag'] = diag; json.dump(out, open(os.path.expanduser(('~/spore_v5/d4/out/dl_%s_%d_%s_%d' + ('_s100' if os.environ.get('S100')=='1' else '') + '.json') % (T.ENV, T.NB, os.environ.get('TAUS', str(T.TAU)), ts)), '

## d3/g6d.py -> d4/code/g6d.py
< os.environ.setdefault('OMP_NUM_THREADS', '1'); HERE = os.path.expanduser('~/spore_v5/a31/spores_2'); sys.path[:0] = [os.path.join(HERE, 'bench'), os.path.join(HERE, 'v6'), os.path.dirname(os.path.abspath(__file__))]
< import numpy as np, common, envs_6d
> os.environ.setdefault('OMP_NUM_THREADS', '1'); HERE = os.path.expanduser('~/spore_v5/d4/spores_2'); sys.path[:0] = [os.path.join(HERE, 'bench'), os.path.join(HERE, 'v6'), os.path.dirname(os.path.abspath(__file__))]
> import numpy as np, common, envs_6d
<         out['res'].append(r); print(r, flush=True)
>         out['res'].append(dict(r, us=[[float(v) for v in u] for u in us], dts=[float(t) for t in dts])); print(r, flush=True)
<     os.makedirs(os.path.expanduser('~/spore_v5/d3/out'), exist_ok=True); json.dump(out, open(os.path.expanduser('~/spore_v5/d3/out/g6d_%s_%d_%d%s.json' % (ENVN, NB, SEED, os.environ.get('TAG', ''))), 'w'), default=float)
>     os.makedirs(os.path.expanduser('~/spore_v5/d4/out'), exist_ok=True); json.dump(out, open(os.path.expanduser('~/spore_v5/d4/out/g6d_%s_%d_%d%s.json' % (ENVN, NB, SEED, os.environ.get('TAG', ''))), 'w'), default=float)

## ../../../bench/refine_ca.py -> d4/code/refine_ca_d4.py
<             if T >= best[0] or np.any(d < -1e-9): continue
>             if T >= best[0] or np.any(d < -float(os.environ.get("CADNEG", 1e-9))): continue
