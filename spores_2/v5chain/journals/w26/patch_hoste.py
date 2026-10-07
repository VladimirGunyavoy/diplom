import sys
p = sys.argv[1]; t = open(p, encoding='utf-8').read()
a = "gX = torch.as_tensor(IDX, device=dev); gW = torch.as_tensor(W, device=dev)"
b = "gX = torch.as_tensor(IDX, device='cpu' if HOSTE else dev); gW = torch.as_tensor(W, device='cpu' if HOSTE else dev)"
c = "vi = Vt[gX[a:a + ch].long()]; w_ = gW[a:a + ch].double()"
d = "vi = Vt[gX[a:a + ch].to(dev).long()]; w_ = gW[a:a + ch].to(dev).double()"
assert t.count(a) == 1 and t.count(c) == 1, (t.count(a), t.count(c))
t = t.replace(a, b).replace(c, d)
k = "        if int(E('SOLVEGPU', 0)):"
assert t.count(k) == 1
t = t.replace(k, "        HOSTE = int(E('HOSTE', 0))   # w26: IDX/W остаются в ОЗУ, на GPU — чанками по проходу (граф 4D dp1 ≥ 16 ГБ не влезает целиком)\n" + k)
open(p, 'w', encoding='utf-8').write(t); print('ok', p)
