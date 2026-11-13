import sys
p = sys.argv[1]; s = open(p).read()
a = "            _R.append((k, S.stats()['return_status'], int(S.stats()['iter_count']), round(time.perf_counter() - t0, 4)))\n"
assert s.count(a) == 1 and 'CUTINF' not in s
s = s.replace(a, a + "            if os.environ.get('CUTINF') and k == 0 and S.stats()['return_status'] == 'Infeasible_Problem_Detected': _L.append((k, 'fail')); break      # блок 9: infeasible на попытке 0 — без повторов\n")
open(p, 'w').write(s)
