"""склейка эталона: OUT/car_q<i>.npy → starts100.npy (100×5: x,y,θ,v,T_ref; T_ref=nan, если не сошёлся). Запуск: python3 merge.py OUT_N80 [OUT_N40 …] (эталон = min по N; первая папка — куда пишется starts100.npy)"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import ref_car as R
Os = sys.argv[1:]; O = Os[0]; Q = R.starts(100, 1000); out = np.full((100, 5), np.nan)
for i, x0 in enumerate(Q):
    out[i, :4] = x0; Ts = [np.load(os.path.join(d, 'car_q%d.npy' % i))[0] for d in Os if os.path.exists(os.path.join(d, 'car_q%d.npy' % i))]
    if Ts: out[i, 4] = min(Ts)   # эталон = min по N (static_car.md)
np.save(os.path.join(O, 'starts100.npy'), out); print('эталон сошёлся', int(np.isfinite(out[:, 4]).sum()), '/ 100'); print('мед T_ref %.3f' % np.nanmedian(out[:, 4]))
