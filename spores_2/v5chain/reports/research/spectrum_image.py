"""research hub-research-6: образ коробки U за τ под потоком (постоянное u). Ошибка интерполяции образа по вершинам/слоям vs τ.
Модели: (A) по каналам ±1 + дрейф (2m+1 слоёв): Z(u) = φ0 + Σ u_i b_i + Σ u_i² c_i; (B) по вершинам коробки (2^m) — мультилинейная (+дрейф для m=1: квадратичная = A).
Ошибка — max по сетке u ∈ U ||φ_τ(x,u) − Z(u)|| (евклид в координатах системы)."""
import numpy as np, itertools

def rk4(f, x, u, t, n=200):
    h = t / n
    for _ in range(n):
        k1 = f(x, u); k2 = f(x + h/2*k1, u); k3 = f(x + h/2*k2, u); k4 = f(x + h*k3, u); x = x + h/6*(k1+2*k2+2*k3+k4)
    return x

pend = lambda x, u: np.array([x[1], -np.sin(x[0]) + u[0]])
dd = lambda x, u: np.array([u[0]*np.cos(x[2]), u[0]*np.sin(x[2]), u[1]])            # (v, ω), квадрат U
def dpend(x, u, g=1.0):
    """двойной маятник, точечные массы 1 на концах, l=1; q от вертикали вниз; τ = u (2 мотора)."""
    q1, q2, w1, w2 = x; c = np.cos(q2); s = np.sin(q2)
    M = np.array([[3 + 2*c, 1 + c], [1 + c, 1.0]])
    C = np.array([-s*(2*w1*w2 + w2**2), s*w1**2])
    G = g*np.array([2*np.sin(q1) + np.sin(q1+q2), np.sin(q1+q2)])
    a = np.linalg.solve(M, np.asarray(u) - C - G)
    return np.array([w1, w2, a[0], a[1]])

def errs(f, x, umax, tau, m, ng=9):
    U = np.linspace(-1, 1, ng); grid = list(itertools.product(U, repeat=m))
    phi = lambda u: rk4(f, x, umax*np.asarray(u, float), tau)
    p0 = phi(np.zeros(m)); e = np.eye(m)
    pp = [phi(e[i]) for i in range(m)]; pm = [phi(-e[i]) for i in range(m)]
    b = [(pp[i]-pm[i])/2 for i in range(m)]; c = [(pp[i]+pm[i])/2 - p0 for i in range(m)]
    V = {s: phi(np.array(s, float)) for s in itertools.product((-1, 1), repeat=m)}
    eA = eB = eL = 0.0
    for u in grid:
        u = np.array(u); ex = phi(u)
        zl = p0 + sum(u[i]*b[i] for i in range(m))                                   # 1-й порядок: параллелепипед
        zA = zl + sum(u[i]**2*c[i] for i in range(m))
        w = {s: np.prod([(1+si*ui)/2 for si, ui in zip(s, u)]) for s in V}           # мультилинейная по вершинам
        zB = sum(w[s]*V[s] for s in V)
        eL, eA, eB = max(eL, np.linalg.norm(ex-zl)), max(eA, np.linalg.norm(ex-zA)), max(eB, np.linalg.norm(ex-zB))
    size = max(np.linalg.norm(b[i]) for i in range(m))
    return eL, eA, eB, size

cases = [('маятник низ θ=0 ω=0 u=.3', pend, np.array([0., 0.]), .3, 1),
         ('маятник θ=asin(.3) (держит +u)', pend, np.array([np.arcsin(.3), 0.]), .3, 1),
         ('маятник θ=2 ω=1', pend, np.array([2., 1.]), .3, 1),
         ('дифдрайв квадрат (v,ω)', dd, np.array([0., 0., 0.]), 1.0, 2),
         ('дв.маятник g=1 низ', dpend, np.array([0., 0., 0., 0.]), 1.0, 2),
         ('дв.маятник g=1 q=(1,-.5) w=(.5,1)', dpend, np.array([1., -.5, .5, 1.]), 1.0, 2)]
for name, f, x, um, m in cases:
    print(name)
    for tau in (.1, .2, .4, .8):
        eL, eA, eB, s = errs(f, x, um, tau, m)
        print(f'  τ={tau:.1f}  |b|={s:.3e}  lin={eL:.2e} ({eL/s:.1e})  chan2m+1={eA:.2e}  vert2^m={eB:.2e}')
