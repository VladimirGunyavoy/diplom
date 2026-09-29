"""Маятник v6: V по сетке, Q-жадный агент до верха. Запуск: python3 tests/test_atlas6_pend.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.pend import PendAtlas

def test_swingup():
    P = PendAtlas(); it = P.solve()
    reach = float(np.mean(P.V < 1e2))
    print(f"итераций {it}, доля достижимых узлов {reach:.3f}, V(низ, покой) = {P.value((0.0, 0.0)):.2f}, V(верх−0.7, 0) = {P.value((np.pi - 0.7, 0.0)):.2f}")
    assert reach > 0.9 and P.value((0.0, 0.0)) > P.value((np.pi - 0.7, 0.0))
    starts = [(0.0, 0.0), (0.5, 0.0), (-1.0, 1.0), (2.0, -1.0), (0.1, 2.0), (-2.5, 0.0)]
    ok = 0
    for s in starts:
        Tt, sw = P.run_agent(s)
        print(f"  старт {s}: V={P.value(s):.2f}, агент {Tt if Tt is None else round(Tt, 2)} с, переключений {sw}")
        ok += Tt is not None
    assert ok >= 5

if __name__ == "__main__":
    test_swingup(); print("ok test_swingup")
