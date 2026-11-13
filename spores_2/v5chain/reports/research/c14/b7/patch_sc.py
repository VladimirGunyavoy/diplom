import sys
for p in sys.argv[1:]:
    s = open(p).read(); a = "S = SysN(fl, 2 ** n,"
    if "flow_rows" in s: continue
    i = s.index(a); j = s.index("\n", s.index("per=(2 * np.pi,) * n + (0,) * n", i))
    s = s[:j] + "\nif os.environ.get('FASTREPLAY'):\n    import fast_flow; S.flow_rows = fast_flow.make(e.G, n)" + s[j:]; open(p, 'w').write(s)
