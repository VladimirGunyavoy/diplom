"""Коридор маятника: агент → сегменты → SLSQP; время до (π,0) против V и агента. Запуск: python3 tests/test_atlas6_gcorridor.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from src.atlas6.pend import PendAtlas
from src.atlas6.gcorridor import agent_trace, segments, endpoint, optimize

def test_corridor():
    P = PendAtlas(); P.solve()
    for s in [(0.0, 0.0), (-1.0, 1.0), (2.0, -1.0), (-2.5, 0.0)]:
        us, xe, ok = agent_trace(P, s)
        assert ok
        seg = segments(us); T_agent = 0.02 * len(us)
        target = (np.round((xe[0] - np.pi) / (2 * np.pi)) * 2 * np.pi + np.pi, 0.0)            # (π, 0) в той же ветке θ, что у агента
        e0 = float(np.hypot(*(endpoint(P.f, s, seg) - np.array(target))))
        cor, T, res = optimize(P.f, s, seg, target)
        print(f"старт {s}: V={P.value(s):.2f}, агент {T_agent:.2f} с ({len(seg)} сегм.), промах до/после {e0:.2f}/{res:.1e}, коридор {T:.2f} с, знаков u: {[int(np.sign(u)) for _, u in cor]}")
        assert res < 1e-5

if __name__ == "__main__":
    test_corridor(); print("ok test_corridor")
