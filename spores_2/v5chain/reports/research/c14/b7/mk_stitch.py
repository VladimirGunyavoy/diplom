import sys
s = open(sys.argv[1]).read()
s = s.replace("from src.atlas6.corridor_nd import refine, merge", "from src.atlas6 import corridor_nd as CN\nfrom src.atlas6.corridor_nd import merge\nORIG = None\ndef refine(*a, **k): return CN.refine(*a, **k)")
a = "T, sq, d, ok = refine(flow, x0, path, g, tries=tries, wlim=wlim); return T, sq, d, ok"
assert a in s; s = s.replace(a, "T, sq, d, ok = (ORIG or CN.refine)(flow, x0, path, g, tries=tries, wlim=wlim); return T, sq, d, ok")
open(sys.argv[2], 'w').write(s)
