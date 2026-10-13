"""viz (линия B): рисовалки дифдрайва и машины. pygame + numpy. Экспорт SYSTEMS для play (образец — draw_links.py): draw(surf, cam, p, x, u, t), static(surf, cam, p, data), trail(p, x).
Препятствия (диски) и сетку рисует ядро; здесь — тело системы и цель."""
import math
import numpy as np, pygame

INK = (30, 34, 44); BODY = (38, 98, 190); BODY_DK = (24, 62, 130); WHEEL = (40, 40, 48); NOSE = (255, 196, 40); GOALC = (60, 170, 90)

def poly(surf, cam, pts, col, edge=None, w=2):
    q = [cam.pt((float(a), float(b))) for a, b in pts]; pygame.draw.polygon(surf, col, q)
    if edge: pygame.draw.polygon(surf, edge, q, w)
def rot(pts, th, c):
    ct, st = math.cos(th), math.sin(th); return [(c[0] + a * ct - b * st, c[1] + a * st + b * ct) for a, b in pts]

# ---------- дифдрайв: состояние (x, y, θ), u = (v, ω), |v| + |ω| ≤ 1 ----------
def dd_draw(surf, cam, p, x, u, t):
    """круглое тело, два колеса по бокам (штрих длиннее — колесо быстрее), жёлтый нос по курсу"""
    r = p.get('body_r', .2); L = p.get('wheel_base', .34) / 2; px, py, th = float(x[0]), float(x[1]), float(x[2])
    v, w = (float(u[0]), float(u[1])) if len(u) >= 2 else (0., 0.); vl, vr = v - w * L, v + w * L
    for side, vw in ((1, vl), (-1, vr)):                                           # колесо слева (+) / справа (−) в системе тела
        hl = r * .45 + r * .35 * min(abs(vw) / .5, 1.5); poly(surf, cam, rot([(-hl, side * L - r * .13), (hl, side * L - r * .13), (hl, side * L + r * .13), (-hl, side * L + r * .13)], th, (px, py)), WHEEL)
    c = cam.pt((px, py)); pygame.draw.circle(surf, BODY, c, cam.len(r)); pygame.draw.circle(surf, BODY_DK, c, cam.len(r), 2)
    poly(surf, cam, rot([(r * .95, 0), (r * .35, r * .32), (r * .35, -r * .32)], th, (px, py)), NOSE, BODY_DK, 1)
def dd_static(surf, cam, p, data):
    """цель — окно ±.05 по x, y и стрелка θ = 0 (ориентация цели)"""
    g = data.get('goal', {}); c = g.get('c', [0, 0]); hx, hy = g.get('half', [.05, .05])[:2]; a = cam.pt((c[0] - hx, c[1] + hy)); b = cam.pt((c[0] + hx, c[1] - hy))
    pygame.draw.circle(surf, (190, 235, 205), cam.pt(c), cam.len(.14)); pygame.draw.circle(surf, GOALC, cam.pt(c), cam.len(.14), 2); pygame.draw.rect(surf, GOALC, pygame.Rect(a[0], a[1], max(3, b[0] - a[0]), max(3, b[1] - a[1])))
    pygame.draw.line(surf, GOALC, cam.pt(c), cam.pt((c[0] + .3, c[1])), 3)

# ---------- машина: состояние (x, y, θ, v), u = (a, ω) ----------
def car_draw(surf, cam, p, x, u, t):
    """кузов-прямоугольник, окна, стоп-сигналы горят при торможении (a < 0), фары — при разгоне"""
    px, py, th = float(x[0]), float(x[1]), float(x[2]); Lc, Wc = p.get('car_len', .8), p.get('car_wid', .44); a = float(u[0]) if len(u) else 0.
    poly(surf, cam, rot([(-Lc / 2, -Wc / 2), (Lc / 2, -Wc / 2), (Lc / 2, Wc / 2), (-Lc / 2, Wc / 2)], th, (px, py)), BODY, BODY_DK, 2)
    poly(surf, cam, rot([(Lc * .12, -Wc * .38), (Lc * .34, -Wc * .38), (Lc * .34, Wc * .38), (Lc * .12, Wc * .38)], th, (px, py)), (190, 220, 245))
    brk = (255, 60, 50) if a < -.05 else (120, 40, 40); head = (255, 236, 150) if a > .05 else (200, 200, 190)
    for sd in (-1, 1):
        poly(surf, cam, rot([(-Lc / 2, sd * Wc * .30), (-Lc / 2 + Lc * .08, sd * Wc * .30), (-Lc / 2 + Lc * .08, sd * Wc * .44), (-Lc / 2, sd * Wc * .44)], th, (px, py)), brk)
        poly(surf, cam, rot([(Lc / 2 - Lc * .08, sd * Wc * .30), (Lc / 2, sd * Wc * .30), (Lc / 2, sd * Wc * .44), (Lc / 2 - Lc * .08, sd * Wc * .44)], th, (px, py)), head)
def car_static(surf, cam, p, data):
    g = data.get('goal', {}); c = g.get('c', [4, 3.6]); r = g.get('r', .45)
    pygame.draw.circle(surf, (190, 235, 205), cam.pt(c), cam.len(r)); pygame.draw.circle(surf, GOALC, cam.pt(c), cam.len(r), 3)
    for k in range(4):                                                              # финишная «шашка»
        a = math.pi / 2 * k; pygame.draw.line(surf, GOALC, cam.pt((c[0] + .3 * r * math.cos(a), c[1] + .3 * r * math.sin(a))), cam.pt((c[0] + .8 * r * math.cos(a), c[1] + .8 * r * math.sin(a))), 2)

SYSTEMS = {
    'dd': dict(draw=dd_draw, static=dd_static, trail=lambda p, x: [(x[0], x[1])]),
    'ddobst': dict(draw=dd_draw, static=dd_static, trail=lambda p, x: [(x[0], x[1])]),
    'car': dict(draw=car_draw, static=car_static, trail=lambda p, x: [(x[0], x[1])]),
    'carfree': dict(draw=car_draw, static=car_static, trail=lambda p, x: [(x[0], x[1])]),
}
