import sys
p = sys.argv[1]; s = open(p).read()
a = "            s.st[S.stats()['return_status']] = s.st.get(S.stats()['return_status'], 0) + 1\n"
assert s.count(a) == 1
s = s.replace(a, a + "            _R.append((k, S.stats()['return_status'], int(S.stats()['iter_count']), round(time.perf_counter() - t0, 4)))\n")
s = s.replace("best = (np.inf, None); US = None; _L = []", "best = (np.inf, None); US = None; _L = []; _R = []")
b = "dict(tries0=tries0, n=n, ok=ok, L=_L, pid=os.getpid())"
assert b in s; s = s.replace(b, "dict(tries0=tries0, n=n, ok=ok, L=_L, R=_R, pid=os.getpid())")
open(p, 'w').write(s)
