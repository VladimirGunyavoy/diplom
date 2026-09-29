"""Обратный Дейкстра от цели (АТЛАС §4 п.5)."""
import heapq
import numpy as np
from collections import defaultdict


def cost_to_go(edges, goal):
    rev = defaultdict(list)
    for k, lst in edges.items():
        for k2, cost, m in lst:
            rev[k2].append((k, cost, m))
    goals = goal if isinstance(goal, list) else [goal]   # ключ — tuple; list — цель-множество
    T, policy = {g: 0.0 for g in goals}, {}
    pq = [(0.0, repr(g), g) for g in goals]
    heapq.heapify(pq)
    while pq:
        t, _, k = heapq.heappop(pq)
        if t > T[k]:
            continue
        for k0, cost, m in rev[k]:
            if t + cost < T.get(k0, np.inf):
                T[k0] = t + cost
                policy[k0] = (m, k)
                heapq.heappush(pq, (t + cost, repr(k0), k0))
    return T, policy
