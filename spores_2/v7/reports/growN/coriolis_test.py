"""п.35 (b4): сохранение энергии плоского 2-звенника при τ = 0 (rk4 dt .001, 3 с, 200 случайных состояний): |ΔE|/E. SYS=manip python3 coriolis_test.py"""
import sys, os, numpy as np
os.environ.setdefault('SYS', 'manip'); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../src/cells7'))
import growN as G
rng = np.random.default_rng(0); x = np.c_[rng.uniform(-3, 3, (200, 2)), rng.uniform(-2, 2, (200, 2))]; z = np.zeros(2)
def F(x): return np.concatenate([x[..., 2:], G.accel(x, z)], -1)
def En(x): c = np.cos(x[:, 1]); m11, m12, m22 = G.AA + 2 * G.BB * c, G.DD + G.BB * c, G.DD; return .5 * (m11 * x[:, 2] ** 2 + 2 * m12 * x[:, 2] * x[:, 3] + m22 * x[:, 3] ** 2)
E0 = En(x); h = .001
for _ in range(3000): k1 = F(x); k2 = F(x + h / 2 * k1); k3 = F(x + h / 2 * k2); k4 = F(x + h * k3); x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
print('max |ΔE|/E %.3g' % np.max(np.abs(En(x) - E0) / E0), 'медиана %.3g' % np.median(np.abs(En(x) - E0) / E0))
