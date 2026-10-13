"""линия E: рисовалки двузвенных систем — dp1 (двойной маятник, g), manip (2 звена, без g), manipf (+ стол и запрет самостолкновения). pygame + numpy.
Состояние x = (q1 − q_off1, q2, ω1, ω2): q1 — угол первого звена от горизонтали, q2 — относительный угол локтя (params['q_off'] = [π/2, 0] у dp1: там y = q − π/2, цель «вверх»).
u = (τ1, τ2): момент в плече и в локте (дуга у сустава — направление и размер ∝ u). Стол (полуплоскость) и сетку рисует ядро; запрет самостолкновения рисуется здесь красным сектором у локтя."""
import math
import numpy as np, pygame
INK = (30, 34, 44); RODC = (60, 70, 90); GOALC = (60, 170, 90); TORQ = (240, 160, 40); BAD = (232, 120, 120)

def aaline(s, col, a, b, w=5):
    pygame.draw.line(s, col, a, b, w); pygame.draw.circle(s, col, a, w // 2); pygame.draw.circle(s, col, b, w // 2)
def pts_of(p, x):
    l1, l2 = p.get('links', [1., 1.]); off = p.get('q_off', [0., 0.]); a1 = x[0] + off[0]; a2 = a1 + x[1] + off[1]
    el = (l1 * math.cos(a1), l1 * math.sin(a1)); return (0., 0.), el, (el[0] + l2 * math.cos(a2), el[1] + l2 * math.sin(a2)), a1, a2
def torque_arc(surf, cam, c, ang, r_u, rad=.26):
    """дуга момента вокруг сустава c (экранные коорд.): от направления ang, длина ∝ r_u ∈ [−1, 1], положительный — против часовой"""
    if abs(r_u) < 1e-2: return
    n = 14; sweep = math.radians(100) * r_u; a0 = ang - sweep / 2; R = cam.len(rad)
    pts = [(c[0] + R * math.cos(a0 + sweep * k / n), c[1] - R * math.sin(a0 + sweep * k / n)) for k in range(n + 1)]
    pygame.draw.lines(surf, TORQ, False, pts, 4); pygame.draw.circle(surf, TORQ, pts[-1], 5)

def arm_draw(surf, cam, p, x, u, t):
    base, el, tip, a1, a2 = pts_of(p, x); b, e, tp = cam.pt(base), cam.pt(el), cam.pt(tip)
    if 'fold' in p:                                                                  # запрещённый сектор локтя: |q2| > π − fold (звено 2 почти сложено назад на звено 1)
        f = p['fold']; R = cam.len(.5); c0 = a1 + math.pi; pts = [e] + [(e[0] + R * math.cos(c0 + f * (2 * k / 12 - 1)), e[1] - R * math.sin(c0 + f * (2 * k / 12 - 1))) for k in range(13)]
        bad = abs((x[1] + math.pi) % (2 * math.pi) - math.pi) > math.pi - f
        pygame.draw.polygon(surf, (240, 150, 150) if bad else (248, 218, 218), pts)
    aaline(surf, RODC, b, e, 11); aaline(surf, RODC, e, tp, 9)
    pygame.draw.circle(surf, (240, 242, 248), e, cam.len(.07)); pygame.draw.circle(surf, INK, e, cam.len(.07), 2)
    pygame.draw.circle(surf, (214, 76, 38), tp, cam.len(.1)); pygame.draw.circle(surf, INK, tp, cam.len(.1), 2)
    pygame.draw.circle(surf, (200, 204, 214), b, cam.len(.13)); pygame.draw.circle(surf, INK, b, cam.len(.13), 3)
    um = p.get('umax', [1., 1.]); r = [float(np.clip(u[k] / um[min(k, len(um) - 1)], -1, 1)) if len(u) > k else 0. for k in (0, 1)]
    torque_arc(surf, cam, b, a1, r[0]); torque_arc(surf, cam, e, a2, r[1], .22)

def arm_static(surf, cam, p, data):
    q = data.get('goal', {}).get('q')
    if q is not None:
        _, el, tip, _, _ = pts_of(p, np.array([q[0] - p.get('q_off', [0, 0])[0], q[1] - p.get('q_off', [0, 0])[1], 0., 0.]))
        b, e, t_ = cam.pt((0, 0)), cam.pt(el), cam.pt(tip); pygame.draw.line(surf, (150, 215, 175), b, e, 8); pygame.draw.line(surf, (150, 215, 175), e, t_, 8)
        pygame.draw.circle(surf, (190, 235, 205), t_, cam.len(.3)); pygame.draw.circle(surf, GOALC, t_, cam.len(.3), 3)
    R = sum(p.get('links', [1., 1.])); pygame.draw.circle(surf, (222, 226, 234), cam.pt((0, 0)), cam.len(R), 1)
    if p.get('gravity'):                                                              # стрелка g в левом верхнем углу области
        a = cam.pt((p['view'][0] + .35, p['view'][3] - .3)); b = (a[0], a[1] + cam.len(.5)); pygame.draw.line(surf, (120, 124, 140), a, b, 3); pygame.draw.polygon(surf, (120, 124, 140), [b, (b[0] - 7, b[1] - 11), (b[0] + 7, b[1] - 11)])
        surf.blit(pygame.font.SysFont('dejavusans,arial', 17).render('g', True, (120, 124, 140)), (a[0] + 9, a[1] + 8))
    if 'table_y' in p:
        f = pygame.font.SysFont('dejavusans,arial', 16); surf.blit(f.render('стол', True, (110, 114, 130)), cam.pt((p['view'][0] + .3, p['table_y'] - .3)))
def arm_trail(p, x):
    return [pts_of(p, x)[2]]

SYSTEMS = {k: dict(draw=arm_draw, static=arm_static, trail=arm_trail) for k in ('dp1', 'manip', 'manipf')}

# present-режим маятника (слева маятник, справа фазовый портрет, SPORE против SST): подмешиваем в запись pend из draw_links ключ 'present'
try:
    import draw_links, present_pend
    SYSTEMS['pend'] = dict(draw_links.SYSTEMS['pend'], present=present_pend.run)
except Exception as ex: print('present для pend не подключён:', ex)
