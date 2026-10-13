"""линия laptop-B (№28): двойной интегратор на фазовой плоскости (x, v) — точка-состояние, цель-круг, кривая переключения x = −v|v|/2 тонкой серой. Экспорт SYSTEMS для play.py."""
import numpy as np, pygame
INK = (30, 34, 44); GOALC = (60, 170, 90); GRID = (232, 234, 240); SW = (175, 180, 190)
_F = {}
def _font(sz=20):
    if sz not in _F: _F[sz] = pygame.font.SysFont('dejavusans,arial,liberationsans,segoeui,helvetica', sz)
    return _F[sz]
def di_draw(surf, cam, p, x, u, t):
    c = cam.pt((float(x[0]), float(x[1]))); col = p.get('_col', (214, 76, 38)); dk = p.get('_col_dk', INK); r = 9
    pygame.draw.circle(surf, col, c, r); pygame.draw.circle(surf, dk, c, r, 2)
def di_static(surf, cam, p, data):
    x0, y0, x1, y1 = p.get('view', [-2.6, -2.6, 2.6, 2.6])
    for k in range(int(x0), int(x1) + 1): pygame.draw.line(surf, GRID, cam.pt((k, y0)), cam.pt((k, y1)), 1)
    for k in range(int(y0), int(y1) + 1): pygame.draw.line(surf, GRID, cam.pt((x0, k)), cam.pt((x1, k)), 1)
    pygame.draw.line(surf, (200, 204, 214), cam.pt((x0, 0)), cam.pt((x1, 0)), 2); pygame.draw.line(surf, (200, 204, 214), cam.pt((0, y0)), cam.pt((0, y1)), 2)
    for txt, pos in (('x', (x1 - .1, -.18)), ('v', (.1, y1 - .1))): im = _font(22).render(txt, True, (110, 116, 130)); surf.blit(im, cam.pt(pos))
    vs = np.linspace(y0, y1, 200); pts = [cam.pt((-v * abs(v) / 2, v)) for v in vs if x0 <= -v * abs(v) / 2 <= x1]                # кривая переключения (общая для обоих методов)
    if len(pts) > 1: pygame.draw.lines(surf, SW, False, pts, 2)
    g = data.get('goal', {}); c = g.get('c', [0, 0]); r = g.get('r', .2)
    pygame.draw.circle(surf, (190, 235, 205), cam.pt(c), cam.len(r)); pygame.draw.circle(surf, GOALC, cam.pt(c), cam.len(r), 3)
SYSTEMS = {'di2d': dict(draw=di_draw, static=di_static, trail=lambda p, x: [(x[0], x[1])], ref_on_top=True)}
