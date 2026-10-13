"""линия D: рисовалки систем со звеньями — маятник, двойной маятник, манипуляторы (плоские). Экспорт SYSTEMS для play.py."""
import math
import numpy as np, pygame
INK = (30, 34, 44); RODC = (60, 70, 90); GOALC = (60, 170, 90)
def aaline(s, col, a, b, w=5):
    pygame.draw.line(s, col, a, b, w); pygame.draw.circle(s, col, a, w // 2); pygame.draw.circle(s, col, b, w // 2)
def label_ang(F=None): pass

# ---------- маятник: θ̈ = sin θ + u, θ=0 — вертикально ВВЕРХ; состояние (θ, ω) ----------
def pend_tip(p, x): L = p.get('L', 1.0); return (L * math.sin(x[0]), L * math.cos(x[0]))
def pend_draw(surf, cam, p, x, u, t):
    L = p.get('L', 1.0); piv = cam.pt((0, 0)); tip = cam.pt(pend_tip(p, x)); pygame.draw.circle(surf, (200, 204, 214), piv, cam.len(.07), 2)
    aaline(surf, RODC, piv, tip, 7); pygame.draw.circle(surf, (214, 76, 38), tip, cam.len(.11)); pygame.draw.circle(surf, INK, tip, cam.len(.11), 2); pygame.draw.circle(surf, INK, piv, cam.len(.04))
    r_ = float(np.clip(u[0] / p.get('umax', [1.])[0], -1, 1)) if len(u) else 0.                              # дуга момента у оси: направление и размер ∝ u
    if abs(r_) > 1e-3:
        r = cam.len(.28); a0 = math.radians(90); sweep = -math.radians(80) * r_; n = 14
        pts = [(piv[0] + r * math.cos(a0 + sweep * k / n), piv[1] - r * math.sin(a0 + sweep * k / n)) for k in range(n + 1)]
        pygame.draw.lines(surf, (240, 160, 40), False, pts, 4)
def pend_static(surf, cam, p, data):
    L = p.get('L', 1.0); g = data.get('goal', {}); tol = g.get('theta', .1); top = cam.pt((0, L))
    pygame.draw.circle(surf, (190, 235, 205), top, cam.len(.16)); pygame.draw.circle(surf, GOALC, top, cam.len(.16), 2)
    pygame.draw.circle(surf, (225, 228, 236), cam.pt((0, 0)), cam.len(L), 1)

SYSTEMS = {
    'pend': dict(draw=pend_draw, static=pend_static, trail=lambda p, x: [pend_tip(p, x)]),
    'pend15': dict(draw=pend_draw, static=pend_static, trail=lambda p, x: [pend_tip(p, x)]),
}
