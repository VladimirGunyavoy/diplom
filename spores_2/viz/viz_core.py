"""viz_core — общий плеер визуализаций систем (линия D). Зависит только от pygame + numpy + json.
Модуль системы (draw_links.py / draw_mobile.py) экспортирует SYSTEMS = {имя: dict(draw=…, static=…, trail=…)}:
  draw(surf, cam, params, x, u, t)      — рисует ОДНО состояние (обязательно)
  static(surf, cam, params, data)       — цель/запретные зоны, специфичные для системы (необязательно)
  trail(params, x) -> [(X, Y), ...]     — мировые точки, чей след рисуется (необязательно)
Данные: viz/data/<система>.json (формат — viz/README.md); params['view'] = [xmin, ymin, xmax, ymax] — мировая область окна (ось y вверх).
Клавиши: Пробел пауза · R сначала · ←/→ старт · ↑/↓ скорость · M метод · G призрак эталона · T след/вся/выкл · Esc выход."""
import os, sys, json, math
import numpy as np
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame

GOALMARK = (40, 140, 80)
W, H = 1280, 720; TOP, BOT = 74, 78
BG = (246, 247, 250); INK = (30, 34, 44); GRID = (226, 229, 236); MUTED = (112, 118, 132)
COLORS = {'споры': (214, 76, 38), 'эталон': (38, 112, 214), 'OCP': (38, 112, 214), 'PPO': (46, 150, 90), 'SST': (140, 90, 170)}
def method_color(name):
    for k, c in COLORS.items():
        if k.lower() in name.lower(): return c
    return (90, 90, 90)
def is_ref(name): return ('эталон' in name.lower()) or ('ocp' in name.lower() and 'multi' in name.lower()) or name.lower() in ('ref', 'ocp')

class Cam:
    """мир → экран; ось y мира вверх"""
    def __init__(s, view, area):
        x0, y0, x1, y1 = view; ax, ay, aw, ah = area; s.scale = min(aw / (x1 - x0), ah / (y1 - y0)); s.ox = ax + aw / 2 - s.scale * (x0 + x1) / 2; s.oy = ay + ah / 2 + s.scale * (y0 + y1) / 2; s.view = view
    def pt(s, p): return (int(round(s.ox + s.scale * p[0])), int(round(s.oy - s.scale * p[1])))
    def len(s, m): return max(1, int(round(s.scale * m)))
    def inv(s, px): return ((px[0] - s.ox) / s.scale, (s.oy - px[1]) / s.scale)

def fonts():
    names = 'dejavusans,arial,liberationsans,segoeui,helvetica,verdana'
    return {k: pygame.font.SysFont(names, sz) for k, sz in (('s', 15), ('m', 19), ('l', 26))}

class Episode:
    def __init__(s, e, params, trail_fn):
        s.e = e; s.method = e['method']; s.start = e['start']; s.dt = float(e['dt']); s.X = np.asarray(e['X'], float); s.U = np.asarray(e.get('U', np.zeros((len(s.X), 1))), float)
        s.T = float(e.get('T') or (len(s.X) - 1) * s.dt); s.Tref = e.get('T_ref'); s.ok = bool(e.get('ok', True)); s.t_goal = (len(s.X) - 1) * s.dt
        s.has_tail = bool(e.get('X_tail')) and len(e['X_tail']) > 0
        if s.has_tail:                                                      # доводка до точной цели (finish_tail.py): после входа в окно цели таймер замирает, тело едет до покоя
            Xt = np.asarray(e['X_tail'], float); Ut = np.asarray(e['U_tail'], float) if e.get('U_tail') else np.zeros((len(Xt), s.U.shape[1] if s.U.ndim == 2 else 1))
            s.X = np.vstack([s.X, Xt]); s.U = np.vstack([s.U, Ut.reshape(len(Ut), -1)[:, :s.U.shape[1]]]) if s.U.ndim == 2 and Ut.size else s.U
        s.tmax = (len(s.X) - 1) * s.dt
        s.trail = None
        if trail_fn: s.trail = [np.array(trail_fn(params, x), float) for x in s.X]       # по кадрам: (k, 2) мировых точек
    def state(s, t):
        t = min(max(t, 0.), s.tmax); f = t / s.dt; i = min(int(f), len(s.X) - 1); j = min(i + 1, len(s.X) - 1); a = f - i
        x = s.X[i] if (a == 0 or np.abs(s.X[j] - s.X[i]).max() > 2.0) else (1 - a) * s.X[i] + a * s.X[j]            # скачок > 2 — обёртка угла: без интерполяции
        k = min(i, len(s.U) - 1); return x, s.U[k], i

def draw_obstacles(surf, cam, data, fill=(205, 208, 218), edge=(150, 155, 170), half=(222, 224, 232)):
    for o in data.get('obstacles', []):
        if o['type'] == 'disc': pygame.draw.circle(surf, fill, cam.pt(o['c']), cam.len(o['r'])); pygame.draw.circle(surf, edge, cam.pt(o['c']), cam.len(o['r']), 2)
        elif o['type'] == 'box':
            (x0, y0), (x1, y1) = o['lo'], o['hi']; a = cam.pt((x0, y1)); b = cam.pt((x1, y0)); r = pygame.Rect(a[0], a[1], b[0] - a[0], b[1] - a[1]); pygame.draw.rect(surf, fill, r); pygame.draw.rect(surf, edge, r, 2)
        elif o['type'] == 'halfplane':                                   # n·p ≥ d — запретная сторона закрашена
            n = np.array(o['n'], float); n /= np.linalg.norm(n); d = float(o['d']); t = np.array([-n[1], n[0]]); p0 = n * d; far = 40.
            poly = [p0 + t * far, p0 - t * far, p0 - t * far + n * far, p0 + t * far + n * far]; pygame.draw.polygon(surf, half, [cam.pt(p) for p in poly]); pygame.draw.line(surf, edge, cam.pt(p0 + t * far), cam.pt(p0 - t * far), 2)

def run(sysname, mod, data, png=None, start=None, method=None, nframes=8, gif=None):
    if png: os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.init(); pygame.font.init(); params = data['params']; ctrl = mod
    scr = pygame.display.set_mode((W, H)); pygame.display.set_caption('spores · ' + data.get('title', sysname)); F = fonts()
    cam = Cam(params['view'], (0, TOP, W, H - TOP - BOT)); draw = ctrl['draw']; static = ctrl.get('static'); trail_fn = ctrl.get('trail')
    eps = [Episode(e, params, trail_fn) for e in data['episodes']]; starts = sorted({e.start for e in eps}, key=lambda v: (str(type(v)), v))
    st = dict(si=0, mi=0, t=0., speed=1., pause=False, ghost=True, trail=2 if ctrl.get('full_path') else 1, hold=0.)
    if start is not None and start in starts: st['si'] = starts.index(start)
    def methods_of(si): return [e for e in eps if e.start == starts[si]]
    if method:
        ms = methods_of(st['si']); st['mi'] = next((i for i, e in enumerate(ms) if method.lower() in e.method.lower()), 0)
    def cur(): ms = methods_of(st['si']); return ms[st['mi'] % len(ms)]
    def ghost_of(e):
        if not st['ghost'] or is_ref(e.method): return None
        return next((g for g in methods_of(st['si']) if is_ref(g.method)), None)
    def draw_obst(surf): draw_obstacles(surf, cam, data)
    def draw_trail(surf, e, i, col, alpha=255):
        """режимы st['trail']: 0 выкл · 1 след (пройденное) · 2 вся траектория бледно + пройденное ярко (флаг full_path у системы — по умолчанию)"""
        if not st['trail'] or e.trail is None: return
        for k in range(e.trail[0].shape[0]):
            if st['trail'] == 2:
                allp = [cam.pt(e.trail[j][k]) for j in range(len(e.trail))]
                if len(allp) > 1: pygame.draw.lines(surf, tuple(int(c * .35 + 255 * .65) for c in col), False, allp, 2)
            pts = [cam.pt(e.trail[j][k]) for j in range(0, i + 1)]
            if len(pts) > 1: pygame.draw.lines(surf, col, False, pts, 3 if st['trail'] == 2 else 2)
    def render(t_now):
        scr.fill(BG)
        step = params.get('grid', 0.5); wx0, wy1 = cam.inv((0, TOP)); wx1, wy0 = cam.inv((W, H - BOT))
        for gx in np.arange(math.floor(wx0 / step) * step, wx1 + step, step): pygame.draw.line(scr, GRID, cam.pt((gx, wy0)), cam.pt((gx, wy1)), 1)
        for gy in np.arange(math.floor(wy0 / step) * step, wy1 + step, step): pygame.draw.line(scr, GRID, cam.pt((wx0, gy)), cam.pt((wx1, gy)), 1)
        draw_obst(scr)
        e = cur(); g = ghost_of(e)
        params['goal_q'] = e.e.get('goal_q')                                 # цель, своя у каждого эпизода (m3gdg)
        if static: static(scr, cam, params, data)
        if g is not None:
            x, u, i = g.state(t_now); layer = pygame.Surface((W, H), pygame.SRCALPHA); draw(layer, cam, params, x, u, t_now)
            scr.blit(pygame.mask.from_surface(layer, 40).to_surface(setcolor=(70, 120, 235, 120), unsetcolor=(0, 0, 0, 0)), (0, 0)); draw_trail(scr, g, i, (150, 175, 235))
        x, u, i = e.state(t_now); draw_trail(scr, e, i, method_color(e.method)); draw(scr, cam, params, x, u, t_now)
        # шапка
        pygame.draw.rect(scr, (255, 255, 255), (0, 0, W, TOP - 8)); pygame.draw.line(scr, GRID, (0, TOP - 8), (W, TOP - 8), 1)
        scr.blit(F['l'].render(data.get('title', sysname), True, INK), (16, 8))
        col = method_color(e.method); pygame.draw.circle(scr, col, (26, 54), 7); ratio = ('   T/T_ref = %.3f' % (e.T / e.Tref)) if (e.Tref and e.ok) else ''
        scr.blit(F['m'].render('%s · старт %s · %s' % (e.method, e.start, 'дошёл' if e.ok else 'НЕ ДОШЁЛ'), True, col if e.ok else (200, 40, 40)), (42, 43))
        tt = F['m'].render('t = %.2f / %.2f с%s' % (min(t_now, e.t_goal), e.T, ratio), True, INK); scr.blit(tt, (W - tt.get_width() - 16, 12))
        if e.has_tail and t_now >= e.t_goal: mk = F['m'].render('goal window reached', True, GOALMARK); scr.blit(mk, (W - mk.get_width() - 16, 70))
        if g is not None: sg = F['s'].render('призрак (синий): %s, T = %.2f с' % (g.method, g.T), True, (60, 100, 200)); scr.blit(sg, (W - sg.get_width() - 16, 44))
        # полоски управления
        n = e.U.shape[1] if e.U.ndim == 2 else 1; um = np.array(params.get('umax', [1.] * n), float); bw = 260
        for k in range(n):
            cy = H - BOT + 10 + int(k * min(18, 62 / n)); cx = 150; pygame.draw.rect(scr, (230, 232, 240), (cx, cy, bw, 11), border_radius=3); pygame.draw.line(scr, MUTED, (cx + bw // 2, cy - 2), (cx + bw // 2, cy + 13), 1)
            v = float(np.clip(u[k] / um[min(k, len(um) - 1)], -1, 1)); xx = cx + bw // 2; pygame.draw.rect(scr, col, (min(xx, xx + v * bw / 2), cy, abs(v * bw / 2), 11), border_radius=3)
            scr.blit(F['s'].render('u%d' % (k + 1) + (' ' + params['unames'][k] if 'unames' in params else ''), True, MUTED), (16, cy - 3))
        hint = F['s'].render('Пробел пауза · R сначала · ←/→ старт · ↑/↓ скорость ×%.2g · M метод · G призрак · T след/вся/выкл · Esc' % st['speed'], True, MUTED); scr.blit(hint, (W - hint.get_width() - 16, H - 24))
    if png:
        os.makedirs(png, exist_ok=True); tag = str(sysname); outs = []
        for si in ([st['si']] if start is not None else range(len(starts))):
            st['si'] = si
            for mi, e in enumerate(methods_of(si)):
                if method and method.lower() not in e.method.lower(): continue
                st['mi'] = mi; fr = []
                for k in range(nframes):
                    render(e.tmax * k / max(nframes - 1, 1)); fn = os.path.join(png, '%s_s%s_%s_f%02d.png' % (tag, e.start, ''.join(ch for ch in e.method if ch.isalnum()), k)); pygame.image.save(scr, fn); fr.append(fn)
                cols = 4; rows = math.ceil(nframes / cols); sw, sh = W // 3, H // 3; mont = pygame.Surface((cols * sw, rows * sh)); mont.fill((255, 255, 255))
                for k, fn in enumerate(fr): mont.blit(pygame.transform.smoothscale(pygame.image.load(fn), (sw, sh)), ((k % cols) * sw, (k // cols) * sh))
                mfn = os.path.join(png, '%s_s%s_%s_montage.png' % (tag, e.start, ''.join(ch for ch in e.method if ch.isalnum()))); pygame.image.save(mont, mfn); outs.append(mfn)
        if gif:
            try:
                from PIL import Image
                e = cur(); ims = []
                for k in range(int(e.tmax / 0.04) + 1): render(k * 0.04); ims.append(Image.frombytes('RGB', (W, H), pygame.image.tostring(scr, 'RGB')).resize((W // 2, H // 2)))
                ims[0].save(gif, save_all=True, append_images=ims[1:], duration=40, loop=0)
            except ImportError: print('--gif: нет PIL, пропущено')
        pygame.quit(); return outs
    clock = pygame.time.Clock(); run_ = True
    while run_:
        dtf = clock.tick(60) / 1000.
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: run_ = False
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k == pygame.K_ESCAPE: run_ = False
                elif k == pygame.K_SPACE: st['pause'] = not st['pause']
                elif k == pygame.K_r: st['t'] = 0.; st['hold'] = 0.
                elif k == pygame.K_RIGHT: st['si'] = (st['si'] + 1) % len(starts); st['t'] = 0.
                elif k == pygame.K_LEFT: st['si'] = (st['si'] - 1) % len(starts); st['t'] = 0.
                elif k == pygame.K_UP: st['speed'] = min(4., st['speed'] * 2)
                elif k == pygame.K_DOWN: st['speed'] = max(.25, st['speed'] / 2)
                elif k == pygame.K_m: st['mi'] = (st['mi'] + 1) % len(methods_of(st['si'])); st['t'] = 0.
                elif k == pygame.K_g: st['ghost'] = not st['ghost']
                elif k == pygame.K_t: st['trail'] = (st['trail'] + 1) % 3
        e = cur(); g = ghost_of(e); tend = max(e.tmax, g.tmax if g else 0.)
        if not st['pause']:
            if st['t'] < tend: st['t'] += dtf * st['speed']
            else:
                st['hold'] += dtf
                if st['hold'] > 1.5: st['t'] = 0.; st['hold'] = 0.
        render(st['t']); pygame.display.flip()
    pygame.quit()
