import sys
old = """                for s_ in range(S.L):
                    sel = act & (Sm[:, step] == s_)
                    if sel.any(): X[sel] = S.wrap(S.flow(X[sel], s_, dt))
"""
new = """                if getattr(S, 'flow_rows', None) is not None:       # c14: слой на строку, один пакетный вызов (numba) вместо L вызовов numpy
                    idx = np.nonzero(act)[0]
                    if len(idx): X[idx] = S.wrap(S.flow_rows(X[idx], Sm[idx, step], dt))
                else:
                  for s_ in range(S.L):
                    sel = act & (Sm[:, step] == s_)
                    if sel.any(): X[sel] = S.wrap(S.flow(X[sel], s_, dt))
"""
for p in sys.argv[1:]:
    s = open(p).read(); assert s.count(old) == 1, p; open(p, 'w').write(s.replace(old, new))
p = sys.argv[1].rsplit('/', 1)[0] + '/../../../../bench/spores_8d_bs.py'
