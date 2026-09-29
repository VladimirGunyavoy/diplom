"""Обратный Дейкстра от цели (АТЛАС §4 п.5)."""
import heapq
import numpy as np
from collections import defaultdict


def cost_to_go(edges, goal):
    rev = defaultdict(list)
    for k, lst in edges.items():
        for k2, cost, m in lst:
            rev[k2].append((k, cost, m))
    T, policy = {goal: 0.0}, {}
    pq = [(0.0, goal)]
    while pq:
        t, k = heapq.heappop(pq)
        if t > T[k]:
            continue
        for k0, cost, m in rev[k]:
            if t + cost < T.get(k0, np.inf):
                T[k0] = t + cost
                policy[k0] = (m, k)
                heapq.heappush(pq, (t + cost, k0))
    return T, policy
