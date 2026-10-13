"""viz_present — окно для записи презентации (`play.py <система> --present [--start K]`; ANIM.md). Окно 1260x900, белый фон, без сетки/подсказок.
Все методы эпизода — в одном кадре на общих часах: SPORE толстой янтарной линией, конкуренты своими цветами, подпись у траектории, T в углу.
Вся траектория бледно с первого кадра, пройденная часть ярко; автоцикл 6-12 с (длинные эпизоды ускоряются, множитель — мелко в углу)."""
import os, math
import numpy as np
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame
import viz_core as vc

PW, PH = 1260, 900; TOP = 64
AMBER = (0xE8, 0xA3, 0x17); AMBER_DK = (0xA8, 0x6F, 0x00); COMP = [(0x2E, 0x7D, 0x6B), (0x8F, 0xB9, 0xA8), (0x6B, 0x7C, 0x76), (0x3B, 0x6E, 0xA8), (0x9A, 0x5F, 0x9E)]
INK = (30, 34, 44); FILL = (222, 224, 228); EDGE = (170, 174, 182)
def is_spore(name): return 'спор' in name.lower() or 'spore' in name.lower()
def dark(c): return tuple(int(v * .62) for v in c)
def pale(c): return tuple(int(v * .4 + 255 * .6) for v in c)

TITLES = {'dd': 'Differential drive', 'ddobst': 'Differential drive + obstacles', 'ddobst7': 'Differential drive + obstacles', 'car': 'Car with inertia + obstacles', 'carfree': 'Car with inertia',
          'm3gdg': '6D arm: many goals', 'm3gd': '6D arm', 'm3gu': '6D arm', 'm8': '8D arm', 'm3d': '3D arm', 'manip': '2-link arm', 'manipf': '2-link arm + table + self-collision',
          'di2d': 'Double integrator: phase plane (x, v)', 'pend': 'Pendulum swing-up', 'pend15': 'Pendulum swing-up', 'dp1': 'Double pendulum'}
def en_method(name):
    """имя метода для слайдов (русские имена остаются в обычном режиме)"""
    n = name.lower()
    if 'дуг' in n: return 'reference (arcs)'                                              # аналитический эталон дуг БЕЗ дисков (не OCP)
    if 'спор' in n or 'spore' in n: return 'SPORE'
    if 'sst' in n: return 'SST 60 s' if '60' in n or n.strip() == 'sst' else name
    if 'bang' in n: return 'exact bang-bang'
    if 'ocp' in n or 'эталон' in n: return 'OCP'
    return name.replace(' с', ' s')                                                       # «RRT 60 с» → «RRT 60 s»

def fit_view(eps, data, params, margin=.10):
    """вписать траектории + цель + препятствия (поле ~10%)"""
    P = [np.asarray(t[:, :2]).reshape(-1, 2) for e in eps for t in [np.vstack(e.trail)]]
    pts = [np.vstack(P)]
    for o in data.get('obstacles', []):
        if o['type'] == 'disc': c = np.array(o['c'], float); r = o['r']; pts.append(np.array([c - r, c + r]))
        elif o['type'] == 'box': pts.append(np.array([o['lo'], o['hi']], float))
    g = data.get('goal', {}) or {}
    if 'c' in g: c = np.array(g['c'][:2], float); r = float(g.get('r', .15)); pts.append(np.array([c - r, c + r]))
    A = np.vstack(pts); lo = A.min(0); hi = A.max(0); d = hi - lo; d = np.maximum(d, 1e-3); lo = lo - margin * d; hi = hi + margin * d
    return [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]

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

def run(sysname, mod, data, png=None, start=None, nframes=4, hold=1.0, hide=None, bscale=.6):
    if png: os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.init(); pygame.font.init(); params = data['params']; draw = mod['draw']; static = mod.get('static'); trail_fn = mod.get('trail')
    eps_all = [vc.Episode(e, params, trail_fn) for e in data['episodes']]
    if start is None: start = pick_start(eps_all)
    eps = [e for e in eps_all if e.start == start and not any(h.lower() in e.method.lower() or h.lower() in en_method(e.method).lower() for h in (hide or []))]
    if not eps: raise SystemExit('нет эпизодов со стартом %r' % (start,))
    eps.sort(key=lambda e: ((not is_spore(e.method)) if mod.get('ref_on_top') else is_spore(e.method), e.method))                                  # спор рисуем последними (поверх)
    names = 'dejavusans,arial,liberationsans,segoeui,helvetica,verdana'
    fL = pygame.font.SysFont(names, 30, bold=True); fM = pygame.font.SysFont(names, 22, bold=True); fS = pygame.font.SysFont(names, 20)
    scr = pygame.display.set_mode((PW, PH)) if not png else pygame.Surface((PW, PH))
    pygame.display.set_caption('SPORE · %s · start %s' % (TITLES.get(sysname, data.get('title', sysname)), start))
    params = dict(params); params.pop('note', None)
    for k_ in ('body_r', 'wheel_base', 'car_len', 'car_wid'):                          # №23: тела мельче, чтобы не налезали на старте
        if k_ in params: params[k_] = params[k_] * bscale
    cam = vc.Cam(list(params['view']) if ('links' in params and 'view' in params) else fit_view(eps, data, params), (0, TOP, PW, PH - TOP))
    col = {}; k = 0
    for e in eps:
        if is_spore(e.method): col[id(e)] = AMBER
        else: col[id(e)] = COMP[k % len(COMP)]; k += 1
    tmax = max(e.tmax for e in eps); play = min(max(tmax, 5.), 11.); mult = tmax / play
    chain = params.get('chain'); prelim = bool(params.get('preliminary') or data.get('preliminary') or (chain or {}).get('preliminary') or 'preliminary' in str(data.get('title', '')).lower())
    def seg(e, t_now):
        """цепочка целей (m3gdg): номер текущей цели k и индексы кадров текущего перехода [a, b]; без цепочки — весь путь"""
        if not chain: return None, 0, len(e.trail) - 1
        ts = chain['t_switch']; k = max(j for j, x in enumerate(ts) if x <= t_now + 1e-9); a = int(round(ts[k] / e.dt)); b = int(round(ts[k + 1] / e.dt)) if k + 1 < len(ts) else len(e.trail) - 1
        return k, a, min(b, len(e.trail) - 1)

    def label_pos(e):
        """подпись — рядом с серединой пути, со сдвигом от остальных подписей"""
        pts = [cam.pt(e.trail[j][0]) for j in range(len(e.trail))]; return pts
    def render(t_now):
        scr.fill((255, 255, 255)); vc.draw_obstacles(scr, cam, data, FILL, EDGE, FILL)
        params['goal_q'] = eps[0].e.get('goal_q'); kg = None
        if chain: kg = seg(eps[0], t_now)[0]; params['goal_q'] = chain['goals'][kg]                    # №25: цель ТЕКУЩЕГО перехода (зелёный контур звеньев)
        if static: static(scr, cam, params, data)
        used = []
        for e in eps:
            if not e.ok: fx, fy = cam.pt(e.trail[-1][0]); used.append(pygame.Rect(fx - 14, fy - 14, 28, 28))
        for e in eps:                                                                    # бледные полные пути
            c = col[id(e)]; w = 4 if is_spore(e.method) else 3
            ks, sa, sb = seg(e, t_now); pts = [cam.pt(e.trail[j][0]) for j in range(sa, sb + 1)]
            if len(pts) > 1: thick(scr, pale(c), pts, w)
        for e in eps:                                                                    # пройденное + тело
            c = col[id(e)]; x, u, i = e.state(t_now); w = 8 if is_spore(e.method) else 5
            ks, sa, sb = seg(e, t_now); pts = [cam.pt(e.trail[j][0]) for j in range(sa, min(i, sb) + 1)]
            if len(pts) > 1: thick(scr, c, pts, w)
            params['_col'] = c; params['_col_dk'] = dark(c); draw(scr, cam, params, x, u, t_now)
        params.pop('_col', None); params.pop('_col_dk', None)
        for e in eps:                                                                    # крестик + подпись у траектории
            c = col[id(e)]; pts = [cam.pt(e.trail[j][0]) for j in range(len(e.trail))]
            if not e.ok:
                fx, fy = pts[-1]; pygame.draw.line(scr, dark(c), (fx - 13, fy - 13), (fx + 13, fy + 13), 6); pygame.draw.line(scr, dark(c), (fx - 13, fy + 13), (fx + 13, fy - 13), 6)
            tx = fM.render(en_method(e.method), True, dark(c) if not is_spore(e.method) else AMBER_DK)
            n = len(eps); ki = eps.index(e); fr0 = [.35, .6, .8, .5][ki % 4] if n > 1 else .5; best = None            # №23: перебор позиций вдоль пути и сторон — первая без наложений (≥ 20 px от других подписей/крестиков)
            for f in [fr0] + [.5, .35, .65, .2, .8, .1, .9]:
                ia = min(int(len(pts) * f), len(pts) - 1); ax, ay = pts[ia]; a2 = pts[min(ia + 6, len(pts) - 1)]; a1 = pts[max(ia - 6, 0)]
                tv = np.array([a2[0] - a1[0], a2[1] - a1[1]], float); tv = tv / (np.linalg.norm(tv) or 1.); nv = np.array([-tv[1], tv[0]])
                for sg in ((1, -1) if ki % 2 == 0 else (-1, 1)):
                    dist = 24 + tx.get_height() / 2; rect = tx.get_rect(center=(int(ax + sg * nv[0] * (dist + tx.get_width() / 2 * abs(nv[0]))), int(ay + sg * nv[1] * dist)))
                    ok_ = pygame.Rect(4, TOP, PW - 8, PH - TOP - 4).contains(rect) and not any(rect.inflate(20, 20).colliderect(r) for r in used)
                    if best is None: best = rect
                    if ok_: best = rect; break
                else: continue
                break
            rect = best
            rect.clamp_ip(pygame.Rect(4, TOP, PW - 8, PH - TOP - 4)); used.append(rect.inflate(6, 4)); scr.blit(tx, rect)
        # шапка и столбик T
        scr.blit(fL.render(TITLES.get(sysname, data.get('title', sysname)), True, INK), (18, 14)); 
        sub = fS.render('start %s%s' % (start, ' · ×%.1f' % mult if mult > 1.05 else ''), True, (120, 126, 138)); scr.blit(sub, (PW - sub.get_width() - 18, 8))
        if prelim: pr = fS.render('preliminary', True, (190, 60, 60)); scr.blit(pr, (PW - pr.get_width() - 18, 32))
        if chain: gk = fM.render('goal %d/%d' % (kg + 1, len(chain['goals'])), True, INK); scr.blit(gk, (PW - gk.get_width() - 18, 56))
        y = PH - 18 - 28 * len(eps); box = pygame.Rect(PW - 372, y - 8, 360, 28 * len(eps) + 12)
        for e in sorted(eps, key=lambda e: not is_spore(e.method)):
            c = col[id(e)]; pygame.draw.circle(scr, c, (PW - 352, y + 12), 8)
            if e.ok: tt = '%s — %.2f s' % (en_method(e.method), e.T) + ((' (%.2f)' % (e.T / e.Tref)) if (e.Tref and is_spore(e.method)) else '')
            else: tt = '%s — failed' % en_method(e.method)
            scr.blit(fM.render(tt, True, INK if e.ok else (180, 40, 40)), (PW - 334, y)); y += 28
        ck = fS.render('t = %.1f s' % min(t_now, max(e.t_goal for e in eps)), True, (120, 126, 138)); scr.blit(ck, (18, PH - 30))
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
