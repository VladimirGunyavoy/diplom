"""viz_present — окно для записи презентации (`play.py <система> --present [--start K]`; ANIM.md). Окно 1260x900, белый фон, без сетки/подсказок.
Все методы эпизода — в одном кадре на общих часах: SPORE толстой янтарной линией, конкуренты своими цветами, подпись у траектории, T в углу.
Вся траектория бледно с первого кадра, пройденная часть ярко; автоцикл 6-12 с (длинные эпизоды ускоряются, множитель — мелко в углу)."""
import os, math
import numpy as np
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame
import viz_core as vc

PW, PH = 1260, 900; TOP = 64
AMBER = (0xE8, 0xA3, 0x17); AMBER_DK = (0xA8, 0x6F, 0x00); COMP = [(0x2E, 0x7D, 0x6B), (0x8F, 0xB9, 0xA8), (0x6B, 0x7C, 0x76)]
INK = (30, 34, 44); FILL = (222, 224, 228); EDGE = (170, 174, 182)
def is_spore(name): return 'спор' in name.lower() or 'spore' in name.lower()
def dark(c): return tuple(int(v * .62) for v in c)
def pale(c): return tuple(int(v * .4 + 255 * .6) for v in c)

def thick(surf, col, pts, w):
    """толстая ломаная со скруглёнными стыками (pygame.draw.lines при w>3 рвётся «пунктиром»); близкие точки склеиваются"""
    q = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - q[-1][0]) + abs(p[1] - q[-1][1]) >= 3: q.append(p)
    if len(q) > 1:
        pygame.draw.lines(surf, col, False, q, w)
        for p in q: pygame.draw.circle(surf, col, p, w // 2)

def pick_start(eps):
    """старт, где T/T_ref спор ближе всего к медиане (среди дошедших)"""
    r = {}
    for e in eps:
        if is_spore(e.method) and e.ok and e.Tref: r[e.start] = e.T / e.Tref
    if not r: return eps[0].start
    med = float(np.median(list(r.values()))); return min(r, key=lambda k: abs(r[k] - med))

def run(sysname, mod, data, png=None, start=None, nframes=4, hold=1.0):
    if png: os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.init(); pygame.font.init(); params = data['params']; draw = mod['draw']; static = mod.get('static'); trail_fn = mod.get('trail')
    eps_all = [vc.Episode(e, params, trail_fn) for e in data['episodes']]
    if start is None: start = pick_start(eps_all)
    eps = [e for e in eps_all if e.start == start]
    if not eps: raise SystemExit('нет эпизодов со стартом %r' % (start,))
    eps.sort(key=lambda e: (is_spore(e.method), e.method))                                  # спор рисуем последними (поверх)
    names = 'dejavusans,arial,liberationsans,segoeui,helvetica,verdana'
    fL = pygame.font.SysFont(names, 30, bold=True); fM = pygame.font.SysFont(names, 22, bold=True); fS = pygame.font.SysFont(names, 20)
    scr = pygame.display.set_mode((PW, PH)) if not png else pygame.Surface((PW, PH))
    pygame.display.set_caption('spores · %s · старт %s' % (data.get('title', sysname), start))
    cam = vc.Cam(params['view'], (0, TOP, PW, PH - TOP))
    col = {}; k = 0
    for e in eps:
        if is_spore(e.method): col[id(e)] = AMBER
        else: col[id(e)] = COMP[k % len(COMP)]; k += 1
    tmax = max(e.tmax for e in eps); play = min(max(tmax, 5.), 11.); mult = tmax / play
    prelim = bool(params.get('preliminary') or data.get('preliminary'))

    def label_pos(e):
        """подпись — рядом с серединой пути, со сдвигом от остальных подписей"""
        pts = [cam.pt(e.trail[j][0]) for j in range(len(e.trail))]; return pts
    def render(t_now):
        scr.fill((255, 255, 255)); vc.draw_obstacles(scr, cam, data, FILL, EDGE, FILL)
        params['goal_q'] = eps[0].e.get('goal_q')
        if static: static(scr, cam, params, data)
        used = []
        for e in eps:                                                                    # бледные полные пути
            c = col[id(e)]; w = 4 if is_spore(e.method) else 3
            pts = [cam.pt(e.trail[j][0]) for j in range(len(e.trail))]
            if len(pts) > 1: thick(scr, pale(c), pts, w)
        for e in eps:                                                                    # пройденное + тело
            c = col[id(e)]; x, u, i = e.state(t_now); w = 8 if is_spore(e.method) else 5
            pts = [cam.pt(e.trail[j][0]) for j in range(0, i + 1)]
            if len(pts) > 1: thick(scr, c, pts, w)
            params['_col'] = c; params['_col_dk'] = dark(c); draw(scr, cam, params, x, u, t_now)
        params.pop('_col', None); params.pop('_col_dk', None)
        for e in eps:                                                                    # крестик + подпись у траектории
            c = col[id(e)]; pts = [cam.pt(e.trail[j][0]) for j in range(len(e.trail))]
            if not e.ok:
                fx, fy = pts[-1]; pygame.draw.line(scr, dark(c), (fx - 13, fy - 13), (fx + 13, fy + 13), 6); pygame.draw.line(scr, dark(c), (fx - 13, fy + 13), (fx + 13, fy - 13), 6)
            tx = fM.render(e.method.upper() if is_spore(e.method) else e.method, True, dark(c) if not is_spore(e.method) else AMBER_DK)
            ax, ay = pts[int(len(pts) * .55)]; rect = tx.get_rect(midbottom=(ax, ay - 34))
            for _ in range(12):
                if any(rect.colliderect(r) for r in used): rect.move_ip(0, -rect.height - 4)
            rect.clamp_ip(pygame.Rect(4, TOP, PW - 8, PH - TOP - 4)); used.append(rect.inflate(6, 4)); scr.blit(tx, rect)
        # шапка и столбик T
        scr.blit(fL.render(data.get('title', sysname), True, INK), (18, 14)); 
        sub = fS.render('старт %s%s' % (start, ' · ×%.1f' % mult if mult > 1.05 else ''), True, (120, 126, 138)); scr.blit(sub, (PW - sub.get_width() - 18, 8))
        if prelim: pr = fS.render('preliminary', True, (190, 60, 60)); scr.blit(pr, (PW - pr.get_width() - 18, 32))
        y = PH - 18 - 28 * len(eps); box = pygame.Rect(PW - 372, y - 8, 360, 28 * len(eps) + 12)
        for e in sorted(eps, key=lambda e: not is_spore(e.method)):
            c = col[id(e)]; pygame.draw.circle(scr, c, (PW - 352, y + 12), 8)
            if e.ok: tt = '%s — %.2f с' % (e.method, e.T) + ((' (%.2f)' % (e.T / e.Tref)) if (e.Tref and is_spore(e.method)) else '')
            else: tt = '%s — не дошёл' % e.method
            scr.blit(fM.render(tt, True, INK if e.ok else (180, 40, 40)), (PW - 334, y)); y += 28
        ck = fS.render('t = %.1f с' % min(t_now, tmax), True, (120, 126, 138)); scr.blit(ck, (18, PH - 30))
    if png:
        os.makedirs(png, exist_ok=True); outs = []
        for k in range(nframes):
            render(tmax * (k + 1) / nframes); fn = os.path.join(png, '%s_present_s%s_f%02d.png' % (sysname, start, k)); pygame.image.save(scr, fn); outs.append(fn)
        pygame.quit(); return outs
    clock = pygame.time.Clock(); t = 0.; h = 0.; running = True
    while running:
        dtf = clock.tick(60) / 1000.
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE): running = False
            elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_r: t = 0.; h = 0.
        if t < tmax: t += dtf * mult
        else:
            h += dtf
            if h > hold: t = 0.; h = 0.
        render(t); pygame.display.flip()
    pygame.quit()
