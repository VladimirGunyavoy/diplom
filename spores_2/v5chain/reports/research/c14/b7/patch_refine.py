import sys
p = sys.argv[1]; s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a; s = s.replace(a, b)
rep("best = (np.inf, None); US = flow.__self__ if False else None", "best = (np.inf, None); US = None; _L = []")
rep("if S.stats()['return_status'] not in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'): continue", "if S.stats()['return_status'] not in ('Solve_Succeeded', 'Solved_To_Acceptable_Level'): _L.append((k, 'fail')); continue")
rep("if T >= best[0] or np.any(d < -1e-9): continue", "if T >= best[0] or np.any(d < -1e-9): _L.append((k, 'worse')); continue")
rep("s.stat['rej_goal'] = s.stat.get('rej_goal', 0) + 1; continue", "s.stat['rej_goal'] = s.stat.get('rej_goal', 0) + 1; _L.append((k, 'rej')); continue")
rep("s.stat['rej_w_or_clr'] = s.stat.get('rej_w_or_clr', 0) + 1; continue", "s.stat['rej_w_or_clr'] = s.stat.get('rej_w_or_clr', 0) + 1; _L.append((k, 'rej')); continue")
rep("            best = (T, d)\n        ok = best[1] is not None\n", "            best = (T, d); _L.append((k, 'best', T))\n        ok = best[1] is not None\n        if os.environ.get('TRYLOG'): open(os.environ['TRYLOG'], 'a').write(__import__('json').dumps(dict(tries0=tries0, n=n, ok=ok, L=_L, pid=os.getpid())) + '\\n')\n")
open(p, 'w').write(s)
