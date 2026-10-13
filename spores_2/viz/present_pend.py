"""линия E: `play.py pend --present` — слева маятник, справа фазовый портрет (θ, ω); SPORE против SST 60 с на общих часах (ANIM.md №5).
Окно 1260×900, белый фон, тема брифа (янтарь SPORE, зелёный SST). Берёт эпизоды viz/data/pend.json (OCP-эталон не показывается: на слайде только SPORE и SST). Подключено через SYSTEMS['pend']['present'] в draw_arm2.py."""
import os, math
import numpy as np
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame
import viz_core as vc, viz_present as vp

PW, PH = vp.PW, vp.PH; TOP = 64
INK = vp.INK; MUTED = (120, 126, 138); GRIDC = (232, 234, 238); GOAL = (60, 170, 90)
def wrap(a): return (a + math.pi) % (2 * math.pi) - math.pi

def run(sysname, mod, data, png=None, start=None, nframes=4, hold=1.0):
    if png: os.environ['SDL_VIDEODRIVER'] = 'dummy'
    pygame.init(); pygame.font.init(); params = data['params']; L = params.get('L', 1.0)
    eps_all = [vc.Episode(e, params, None) for e in data['episodes']]
    if start is None: start = vp.pick_start([e for e in eps_all if vp.is_spore(e.method) or 'sst' in e.method.lower()])
    eps = [e for e in eps_all if e.start == start and (vp.is_spore(e.method) or 'sst' in e.method.lower())]
    if not eps: raise SystemExit('нет эпизодов SPORE/SST со стартом %r' % (start,))
    eps.sort(key=lambda e: vp.is_spore(e.method)); col = {id(e): (vp.AMBER if vp.is_spore(e.method) else vp.COMP[0]) for e in eps}
    names = 'dejavusans,arial,liberationsans,segoeui,helvetica,verdana'
    fL = pygame.font.SysFont(names, 30, bold=True); fM = pygame.font.SysFont(names, 22, bold=True); fS = pygame.font.SysFont(names, 20)
    scr = pygame.display.set_mode((PW, PH)) if not png else pygame.Surface((PW, PH))
    pygame.display.set_caption('SPORE · Pendulum swing-up · start %s' % start)
    cam = vc.Cam([-1.45, -1.45, 1.45, 1.55], (10, TOP + 20, 620, 700))                        # левая половина: маятник (θ = 0 — вверх)
    wmax = max(float(np.abs(e.X[:, 1]).max()) for e in eps) * 1.1 + .2; px0, px1, py0, py1 = 700, 1220, TOP + 70, TOP + 70 + 520          # правая: фазовый портрет
    def ph(th, w): return (int(px0 + (wrap(th) + math.pi) / (2 * math.pi) * (px1 - px0)), int(py1 - (w + wmax) / (2 * wmax) * (py1 - py0)))
    tmax = max(e.tmax for e in eps); play = min(max(tmax, 5.), 11.); mult = tmax / play
    um = float(params.get('umax', [1.])[0])

    def tip(x): return (L * math.sin(x[0]), L * math.cos(x[0]))
    def phase_pts(e, upto):
        """ломаные фазовой траектории; на обёртке θ через ±π линия рвётся"""
        segs = [[]]; prev = None
        for j in range(upto + 1):
            th = wrap(e.X[j][0]); p = ph(th, e.X[j][1])
            if prev is not None and abs(th - prev) > math.pi: segs.append([])
            segs[-1].append(p); prev = th
        return segs
    def render(t_now):
        scr.fill((255, 255, 255))
        # --- левая панель: маятник ---
        pv = cam.pt((0, 0)); top = cam.pt((0, L)); pygame.draw.circle(scr, (226, 229, 236), pv, cam.len(L), 1)
        pygame.draw.circle(scr, (190, 235, 205), top, cam.len(.17)); pygame.draw.circle(scr, GOAL, top, cam.len(.17), 3)
        for e in eps:
            c = col[id(e)]; x, u, i = e.state(t_now); tp = cam.pt(tip(x)); w = 12 if vp.is_spore(e.method) else 8
            vp.thick(scr, c, [pv, tp], w); pygame.draw.circle(scr, vp.dark(c), tp, cam.len(.1)); pygame.draw.circle(scr, c, tp, cam.len(.1) - 4)
        pygame.draw.circle(scr, INK, pv, cam.len(.045))
        # --- правая панель: фазовый портрет ---
        pygame.draw.rect(scr, (250, 251, 252), (px0, py0, px1 - px0, py1 - py0)); pygame.draw.rect(scr, (205, 208, 216), (px0, py0, px1 - px0, py1 - py0), 2)
        zx = ph(0., 0.); pygame.draw.line(scr, GRIDC, (px0, zx[1]), (px1, zx[1]), 2); pygame.draw.line(scr, GRIDC, (zx[0], py0), (zx[0], py1), 2)
        g = data.get('goal', {}); th_tol = g.get('theta', .1); pygame.draw.circle(scr, (190, 235, 205), zx, 16); pygame.draw.circle(scr, GOAL, zx, 16, 3)
        scr.blit(fM.render('θ', True, MUTED), (px1 - 22, zx[1] + 6)); scr.blit(fM.render('ω', True, MUTED), (zx[0] + 8, py0 + 4))
        scr.blit(fS.render('−π', True, MUTED), (px0 + 4, py1 + 6)); scr.blit(fS.render('π', True, MUTED), (px1 - 14, py1 + 6)); scr.blit(fS.render('phase portrait', True, MUTED), (px0, py0 - 30))
        for e in eps:                                                                            # бледные полные пути, затем пройденная часть
            c = col[id(e)]
            for s in phase_pts(e, len(e.X) - 1):
                if len(s) > 1: vp.thick(scr, vp.pale(c), s, 4)
        for e in eps:
            c = col[id(e)]; x, u, i = e.state(t_now); w = 7 if vp.is_spore(e.method) else 5
            for s in phase_pts(e, i):
                if len(s) > 1: vp.thick(scr, c, s, w)
            cp = ph(x[0], x[1]); pygame.draw.circle(scr, vp.dark(c), cp, 9); pygame.draw.circle(scr, c, cp, 6)
            end = ph(e.X[-1][0], e.X[-1][1]); tx = fM.render(vp.en_method(e.method), True, vp.AMBER_DK if vp.is_spore(e.method) else vp.dark(c))
            i0 = int(len(e.X) * (.25 if vp.is_spore(e.method) else .4)); a = ph(e.X[i0][0], e.X[i0][1]); scr.blit(tx, (min(max(a[0] + 14, px0 + 6), px1 - tx.get_width() - 6), min(max(a[1] - 34, py0 + 4), py1 - 30)))
        # --- шапка, столбик T, часы ---
        scr.blit(fL.render('Pendulum swing-up', True, INK), (18, 14)); sub = fS.render('start %s%s' % (start, ' · ×%.1f' % mult if mult > 1.05 else ''), True, MUTED); scr.blit(sub, (PW - sub.get_width() - 18, 8))
        y = PH - 18 - 28 * len(eps)
        for e in sorted(eps, key=lambda e: not vp.is_spore(e.method)):
            c = col[id(e)]; pygame.draw.circle(scr, c, (PW - 352, y + 12), 8)
            tt = '%s — %.2f s' % (vp.en_method(e.method), e.T) + ((' (%.2f)' % (e.T / e.Tref)) if (e.Tref and vp.is_spore(e.method)) else '') if e.ok else '%s — failed' % vp.en_method(e.method)
            scr.blit(fM.render(tt, True, INK if e.ok else (180, 40, 40)), (PW - 334, y)); y += 28
        scr.blit(fS.render('t = %.1f s' % min(t_now, tmax), True, MUTED), (18, PH - 30))
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
