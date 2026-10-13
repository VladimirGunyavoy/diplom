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

# ---------- плоские цепочки из n звеньев (3 зв. + g: m3gd/m3gu/m3gdg; 4 зв.: m8): θ_i = Σ q_j от горизонтали, длины params['links'] ----------
def chain_pts(p, x):
    ln = p.get('links', [1., 1., 1.]); th = np.cumsum(x[:len(ln)]); pts = [(0., 0.)]
    for l, a in zip(ln, th): pts.append((pts[-1][0] + l * math.cos(a), pts[-1][1] + l * math.sin(a)))
    return pts
def chain_draw(surf, cam, p, x, u, t):
    pts = chain_pts(p, x); sc = [cam.pt(q) for q in pts]; n = len(sc) - 1
    for k in range(n): aaline(surf, RODC, sc[k], sc[k + 1], 9 - k)
    for k in range(n + 1): pygame.draw.circle(surf, (214, 76, 38) if k == n else (240, 242, 248), sc[k], cam.len(.06 if k < n else .09)); pygame.draw.circle(surf, INK, sc[k], cam.len(.06 if k < n else .09), 2)
    pygame.draw.circle(surf, INK, sc[0], cam.len(.12), 3)
def chain_static(surf, cam, p, data):
    g = data.get('goal', {}); q = p.get('goal_q') or g.get('q')
    if q is not None:
        pts = [cam.pt(v) for v in chain_pts(p, np.array(q, float))]
        for k in range(len(pts) - 1): pygame.draw.line(surf, (120, 200, 150), pts[k], pts[k + 1], 7)
        pygame.draw.circle(surf, (60, 170, 90), pts[-1], cam.len(.12), 3)
    for r in (1., 2., 3.): pygame.draw.circle(surf, (228, 231, 238), cam.pt((0, 0)), cam.len(r), 1) if r <= sum(p.get('links', [1, 1, 1])) else None
SYSTEMS.update({k: dict(draw=chain_draw, static=chain_static, trail=lambda p, x: [chain_pts(p, x)[-1]]) for k in ('m3gd', 'm3gu', 'm3gdg', 'm8')})

# ---------- m3d: 3D двузвенник (основание q0 вокруг вертикали, плечо q1 — тангаж, локоть q2): две проекции — сверху (слева) и сбоку в плоскости руки (справа) ----------
def m3d_geom(p, x):
    l1, l2 = p.get('links', [.5, 1.]); q0, q1, q2 = x[0], x[1], x[2]; r1 = l1 * math.cos(q1); z1 = l1 * math.sin(q1); r2 = r1 + l2 * math.cos(q1 + q2); z2 = z1 + l2 * math.sin(q1 + q2); return q0, (r1, z1), (r2, z2)
def m3d_draw(surf, cam, p, x, u, t):
    q0, e1, e2 = m3d_geom(p, x); cl, cr = p.get('cx', [-1.9, 1.9]); c, s = math.cos(q0), math.sin(q0)
    top = lambda r: cam.pt((cl + r * c, r * s)); side = lambda r, z: cam.pt((cr + r, z))
    pygame.draw.circle(surf, (214, 218, 228), cam.pt((cl, 0)), cam.len(1.5), 1); pygame.draw.line(surf, (214, 218, 228), cam.pt((cr - 1.5, 0)), cam.pt((cr + 1.5, 0)), 1)
    a0, a1, a2 = top(0.), top(e1[0]), top(e2[0]); aaline(surf, RODC, a0, a1, 8); aaline(surf, RODC, a1, a2, 6); pygame.draw.circle(surf, (214, 76, 38), a2, cam.len(.08)); pygame.draw.circle(surf, INK, a0, cam.len(.1), 3)
    b0, b1, b2 = side(0., 0.), side(*e1), side(*e2); aaline(surf, RODC, b0, b1, 8); aaline(surf, RODC, b1, b2, 6); pygame.draw.circle(surf, (214, 76, 38), b2, cam.len(.08)); pygame.draw.circle(surf, INK, b0, cam.len(.1), 3)
_FONT = {}
def _font(sz=20):
    if sz not in _FONT: _FONT[sz] = pygame.font.SysFont('dejavusans,arial,liberationsans,segoeui,helvetica', sz)
    return _FONT[sz]
def m3d_static(surf, cam, p, data):
    cl, cr = p.get('cx', [-1.9, 1.9]); _, e1, e2 = m3d_geom(p, np.zeros(6)); lay = pygame.Surface(surf.get_size(), pygame.SRCALPHA); gc = (60, 190, 110, 120)   # целевая поза (q=0, рука вытянута) — полупрозрачные звенья
    for (a0, a1, a2) in (((cl, 0), (cl + e1[0], 0), (cl + e2[0], 0)), ((cr, 0), (cr + e1[0], e1[1]), (cr + e2[0], e2[1]))):
        pygame.draw.line(lay, gc, cam.pt(a0), cam.pt(a1), 14); pygame.draw.line(lay, gc, cam.pt(a1), cam.pt(a2), 10); pygame.draw.circle(lay, gc, cam.pt(a2), cam.len(.14))
    surf.blit(lay, (0, 0))
    for cx, txt in ((cl, 'вид сверху'), (cr, 'вид сбоку (в плоскости руки)')):
        im = _font(20).render(txt, True, (90, 96, 112)); pos = cam.pt((cx, 1.65)); surf.blit(im, (pos[0] - im.get_width() // 2, pos[1]))
SYSTEMS['m3d'] = dict(draw=m3d_draw, static=m3d_static, trail=lambda p, x: [(p.get('cx', [-1.9, 1.9])[0] + m3d_geom(p, x)[2][0] * math.cos(x[0]), m3d_geom(p, x)[2][0] * math.sin(x[0])), (p.get('cx', [-1.9, 1.9])[1] + m3d_geom(p, x)[2][0], m3d_geom(p, x)[2][1])])
