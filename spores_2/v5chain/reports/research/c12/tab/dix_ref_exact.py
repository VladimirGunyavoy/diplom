"""research-37: эталон ДИ 6D (3 независимые оси, коробка |x|,|v| <= .35) — T* = max по осям точного T_box (di4_ref_exact.tbox).
Старты: rng(SEED) uniform [-2,2]^6, keep = остановочный путь внутри поля по всем осям (правило DI4), первые NK keep.
Выход: di6_ref_<NK>_rho35_s<SEED>.npy — (NK, 10): x1 x2 x3 v1 v2 v3 T1 T2 T3 T*. Запуск: python3 dix_ref_exact.py NK SEED NH (NH=3 ДИ6D, 4 ДИ8D)"""
import sys, numpy as np
sys.argv, (NK, SEED, NH) = sys.argv[:1], (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
import di4_ref_exact as ex
rng = np.random.default_rng(SEED); out = []
while len(out) < NK:
    q = rng.uniform(-2, 2, 2 * NH); x, v = q[:NH], q[NH:]
    if np.abs(x + v * np.abs(v) / 2).max() > 2.5: continue
    Ts = [ex.tbox(x[i], v[i]) for i in range(NH)]; out.append(np.r_[q, Ts, max(Ts)])
A = np.array(out); np.save('di%d_ref_%d_rho35_s%d.npy' % (2 * NH, NK, SEED), A); print('T*: mean %.3f min %.3f max %.3f' % (A[:, -1].mean(), A[:, -1].min(), A[:, -1].max()))
