"""Поле ±(3, 2.5) / буфер ±(6, 3.5): рамки и камера в окне. Windows-питон offscreen; картинки reports/field_*.png. env FULL=1 — полный рост (C), иначе только первый шаг."""
import os, sys, time
os.environ['V8_WINDOW'] = 'offscreen'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
def shot(name):
    for _ in range(3): M.app.step(); G.poll()
    print('shot', name, ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports', name))), flush=True)
P = M.scene_setup.player; G = M.growview; print('FIELD', G.field_text()); M._step('C'); G.draw(M.stepper.snapshot); shot('field_default.png')
P.position = (0, 9, -12); P.camera_pivot.rotation_x = 38; P.rotation_y = 0; shot('field_far.png'); print('SMOKE OK')
