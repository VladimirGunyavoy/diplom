import sys; sys.path.insert(0,'.'); sys.argv=['x']
exec(open('tests/run_faces_pend.py').read().split("# эталон")[0])
ln = lines_graded(.02, 1.0, 400, d0=.05); F = Faces(S, ln, ln, m=4, periodic_x=True, tmax=40.0)
print('lines symmetric', np.allclose(ln, -ln[::-1]))
P = F.P; Pm = -P
# map mirrored points: they must be probes too (symmetric set)
t0,e0,k0,o0 = F.hit(P, 0); t1,e1,k1,o1 = F.hit(Pm, 1)
d = np.abs(t0 - t1); print('t mismatch frac', (d > 1e-6).mean(), 'out mismatch', (o0 != o1).mean(), 'exit mismatch', (np.abs(e0 + e1).max(1) > 1e-6).mean())
bad = np.nonzero((d > 1e-6) | (o0 != o1))[0][:8]
for b in bad: print(P[b], 't0', t0[b], 'o0', o0[b], 'k', k0[b], 'e0', e0[b], '| mirror', Pm[b], 't1', t1[b], 'o1', o1[b], 'k', k1[b], 'e1', e1[b])
