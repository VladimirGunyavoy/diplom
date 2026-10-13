"""ctypes-обёртка C-потока удержания (hold8_c.c): hold_flow_c(Y[B,8], U[B,4], t, rec=False) → (Yend, Yhalf[, rec[B,k,4]])."""
import os, ctypes, subprocess, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); SO = os.path.join(HERE, 'libhold8c.so')
if not os.path.exists(SO) or os.path.getmtime(SO) < os.path.getmtime(os.path.join(HERE, 'hold8_c.c')):
    subprocess.check_call(['gcc', '-O3', '-march=native', '-shared', '-fPIC', os.path.join(HERE, 'hold8_c.c'), '-o', SO, '-lm'])
_L = ctypes.CDLL(SO); P = ctypes.POINTER(ctypes.c_double); _L.hold_flow.argtypes = [ctypes.c_int, P, P, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.c_double, P, P, P]
def init(ms, ln, g): _L.set_params((ctypes.c_double * 4)(*ms), (ctypes.c_double * 4)(*ln), ctypes.c_double(g))
def hold_flow_c(Y, U, t, WM=3., H=.01, rec=False):
    Y = np.ascontiguousarray(Y, float).reshape(-1, 8); U = np.ascontiguousarray(U, float).reshape(-1, 4); B = len(Y); k = max(1, int(round(t / H)))
    Yo = np.empty((B, 8)); Yh = np.empty((B, 8)); R = np.empty((B, k, 4)) if rec else None
    p = lambda a: a.ctypes.data_as(P)
    _L.hold_flow(B, p(Y), p(U), t, k, WM, H, p(Yo), p(Yh), p(R) if rec else None)
    return (Yo, Yh, R) if rec else (Yo, Yh)
