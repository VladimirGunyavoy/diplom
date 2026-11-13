import sys
p = sys.argv[1]; s = open(p).read()
a = "S.flow_all = tree_fast.make_flow_all(name, US_);"
i = s.index(a); j = s.index("\n", i)
s = s[:j] + "\n    if os.environ.get('VECTREE'): import tree_vec8; tree_vec8.install()" + s[j:]; open(p, 'w').write(s)
r = sys.argv[2]; t = open(r).read(); b = "            best = (T, d); _L.append((k, 'best', T))\n"
assert b in t; open(r, 'w').write(t.replace(b, b + "            if os.environ.get('EARLYSTOP'): break\n"))
