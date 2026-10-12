"""src/runlog.py: формат файлов журнала запуска (без Ursina)."""
import os, sys, json, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from src.runlog import RunLog


def _res(n=5):
    rng = np.random.default_rng(0); paths = [np.cumsum(rng.normal(size=(8, 2)), 0) for _ in range(n)]; U = [rng.choice([-1., 0., 1.], 8) for _ in range(n)]
    T = np.array([2., np.nan, 3., np.nan, 5.]); status = np.array(['reached', 'failed', 'reached', 'out_of_area', 'reached'], dtype=object); left = np.array([0, 1, 0, 1, 0], bool)
    return dict(paths=paths, V=[np.ones(8)] * n, U=U, T=T, sw=np.arange(5), X0=rng.normal(size=(5, 2)), status=status, prephase=np.array([0, 0, 0, 1, 1], bool), mark=np.full((5, 2), np.nan), left_work=left)


def test_runlog_files():
    d = tempfile.mkdtemp(); R = RunLog(root=d, tee=False); R.params(extra=dict(layers=[1., -1., 0.])); [R._grow_write(x) for x in ('layer', ' ', '1.0', ' ', 'стоп по покрытию:', ' ', '7', ' клеток, непокрыто 0.0021', '\n', 'seed u=1.0 cells 3 overlap(last 5) 0.123\n')]      # print пишет кусками
    cells = [dict(G=np.zeros((4, 3, 2))) for _ in range(3)]; R.atlas(dict(phase='layer', layers=(1., -1.), layer_k=0, cells=cells, cells_by_layer={})); R.atlas(dict(phase='layer', layers=(1., -1.), layer_k=0, cells=cells, cells_by_layer={}))
    R.vcalc('q', dict(n=100, passes=104, converged=True, t=.2, finite=.74, vmax=7., layers={1.: 3}), 3, .3); R.vcalc('q', dict(n=100), 3, 0., cache='hit'); R.traj('q', _res(), 1.5)
    for k in ('v', 'w', 'mouse move', 'y up', 't'): R.key(k, dict(calc='q'), dict(ms=1.5, vcalc='hit'))
    R.close(); D = R.dir; ls = set(os.listdir(D)); assert {'params.json', 'grow.log', 'atlas.json', 'vcalc.jsonl', 'traj_q.csv', 'traj_summary.jsonl', 'keys.log', 'summary.json'} <= ls, ls
    P = json.load(open(os.path.join(D, 'params.json'))); assert P['layers'] == [1., -1., 0.] and 'git' in P and 'growN' in P and 'env' in P and 'started' in P
    A = json.load(open(os.path.join(D, 'atlas.json')))['layers']['1.0']; assert A['cells'] == 3 and A['nodes'] == 36 and A['rows'] == 12 and A['uncovered'] == .0021 and A['overlap_last'] == .123
    V = [json.loads(l) for l in open(os.path.join(D, 'vcalc.jsonl'))]; assert len(V) == 2 and V[1]['cache'] == 'hit' and V[0]['cache'] == 'miss' and V[0]['mode'] == 'q' and V[0]['passes'] == 104 and V[0]['converged'] is True
    C = open(os.path.join(D, 'traj_q.csv')).read().splitlines(); assert C[0] == 'i,x0,v0,status,left_work_area,prephase,T,switches,share_u+1,share_u-1,share_u0,end_x,end_v' and len(C) == 6 and all(len(r.split(',')) == 13 for r in C)
    S = json.loads(open(os.path.join(D, 'traj_summary.jsonl')).read()); assert S['reached'] == 3 and S['n'] == 5 and S['in_area'] == 4 and S['out_of_area'] == 1 and S['failed'] == 1 and S['left_work_area'] == 2 and S['work']['n'] == 3 and S['work']['reached'] == 3 and S['work']['failed'] == 0; st = [r.split(',')[3] for r in C[1:]]; assert st == ['reached', 'failed', 'reached', 'out_of_area', 'reached'], st and S['mode'] == 'q'
    K = open(os.path.join(D, 'keys.log')).read().splitlines(); assert [l.split('\t')[1] for l in K] == ['v', 't'] and json.loads(K[0].split('\t')[2])['ms'] == 1.5 and json.loads(K[0].split('\t')[2])['vcalc'] == 'hit', K
    I = [json.loads(l) for l in open(os.path.join(d, 'index.jsonl'))]; assert len(I) == 1 and I[0]['run'] == R.name and I[0]['n_vcalc'] == 1 and I[0]['n_vcalc_hits'] == 1 and I[0]['trajs'][0]['reached'] == 3 and 'finished' in I[0]
    R2 = RunLog(root=d, tee=False, stamp=R.name); assert R2.name == R.name + '-2'; R2.close(); I = [json.loads(l) for l in open(os.path.join(d, 'index.jsonl'))]; assert len(I) == 2
