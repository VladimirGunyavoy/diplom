import os, sys
os.environ['V8_WINDOW'] = 'offscreen'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
from ursina import window
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot); G.toggle_common(); G.toggle_traj()
for _ in range(5): M.app.step()
print('aspect', window.aspect_ratio, 'size', window.size)
for m in M.screen_manager._messages:
    e = m._text_entity; t = e.text; L = t.split('\n'); print('MSG', m.name, 'lines', len(L), 'maxcols', max(len(x) for x in L), 'pos', tuple(e.position)[:2], 'width', round(e.width, 3), 'height', round(e.height, 3), 'origin', tuple(e.origin))
    if m.name in ('growview', 'bindings_help'): print('\n'.join(L[:14]))
from src.core import screen_manager as SM
asp = window.aspect_ratio
for m in M.screen_manager._messages:
    e = m._text_entity; L = e.text.split('\n'); assert max(len(x) for x in L) <= SM.WRAP_COLS and e.x + e.width <= asp / 2 + 1e-6, (m.name, max(len(x) for x in L), e.x + e.width, asp / 2)
    if m.bottom: assert e.y + e.height <= 0.5 and e.y >= -0.5, (e.y, e.height)
    if m.name == 'bindings_help': assert e.y - e.height >= -0.5, (e.y, e.height)
print('layout ok at aspect %.2f' % asp)
for a in (1.78, 1.0, .6, 2.4):
    for x0 in (-0.86, 0.23):
        import types; ww = types.SimpleNamespace(aspect_ratio=a); SM.window = ww; x = SM.edge_x(x0); c = SM.fit_cols(x); assert -a / 2 - 1e-9 <= x and x + c * SM.CHAR_W <= a / 2 + 1e-9, (a, x0, x, c); print('aspect %.2f x0=%.2f -> x=%.3f cols=%d right edge %.3f <= %.3f' % (a, x0, x, c, x + c * SM.CHAR_W, a / 2))
SM.window = window
t = SM.wrap_text('k - skeleton: normal / time and a very long explanation of what this key does in the window, with many words', 40); print(t); assert all(len(x) <= 40 for x in t.split('\n')) and t.split('\n')[1].startswith('    ')
for x, a in ((-0.86, 1.78), (-0.86, 1.0), (-0.86, 0.6), (0.23, 1.0)): print('fit_cols x=%.2f aspect=%.1f ->' % (x, a), max(20, min(SM.WRAP_COLS, int((a / 2 - x - 0.02) / SM.CHAR_W))))
P = M.scene_setup.player; P.cursor.enabled = False; P.position = (0, 6, -9); P.camera_pivot.rotation_x = 32
for _ in range(3): M.app.step()
print('shot', ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(os.environ.get('SHOT_DIR', r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports'), os.environ.get('SHOT', 'text_before.png')))))
