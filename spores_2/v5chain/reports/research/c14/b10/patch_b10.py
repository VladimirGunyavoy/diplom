import sys
p = sys.argv[1]; s = open(p).read()
assert '_select_nb' not in s
helper = '''
def _select_nb(S, flow, x0, nb, miss):
    """блок 10: отбор соседей для polish по T до полировки (pre = сумма длительностей + промах конца; как score кандидатов). POLISH=P1 — top-3; P2 — pre <= 1.03 * лучший pre среди соседей."""
    import os
    mode = os.environ.get('POLISH')
    if not mode or not nb: return nb
    pre = []
    for sqn, dn in nb:
        xe = endpoint(flow, x0, sqn, np.asarray(dn, float)); pre.append(float(np.sum(dn)) + float(miss(xe)))
    o = np.argsort(pre, kind='stable'); pre = np.asarray(pre)
    keep = o[:3] if mode == 'P1' else o[pre[o] <= 1.03 * pre[o[0]]]
    return [nb[i] for i in keep]

'''
a = "def endpoint(flow, x0, seq, dts):"
i = s.index(a); j = s.index("\n\n\n", i)
s = s[:j + 1] + helper + s[j + 1:]
b = "        for sqn, dn in nb:\n"
assert s.count(b) == 1
s = s.replace(b, "        nb = _select_nb(S, flow, x0, nb, miss)\n" + b)
open(p, 'w').write(s)
