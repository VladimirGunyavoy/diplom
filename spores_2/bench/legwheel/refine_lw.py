"""линия D: refine коридора через CasADi/IPOPT для робота — подкласс refine_ca.CA (динамика L.f_ca, слои LAYERS, без периодов: tgt=0)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
import numpy as np, casadi as ca
import refine_ca, lw_dyn as L
class CALW(refine_ca.CA):
    def __init__(s, layers, ms=10, maxit=60):
        s.ms = ms; s.maxit = maxit; s.fallback = False; s.orig = None; s.cache = {}; s.st = {}; s.stat = dict(n=0, t=0., it=0); s.n = 3; s.nu = 2; s.C3 = np.zeros(3); s.US = np.array(layers, float)
        x = ca.SX.sym('x', 6); u = ca.SX.sym('u', 2); s.fx = ca.Function('f', [x, u], [L.f_ca(x, u)])
def install(layers, ms=10, maxit=60, prebuild=()):
    from src.atlas6 import corridor_nd
    c = CALW(layers, ms, maxit); c.orig = corridor_nd.refine; [c.solver(n, False, False) for n in prebuild]; corridor_nd.refine = c.refine; return c
