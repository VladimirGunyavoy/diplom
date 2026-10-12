import os, sys
os.environ['V8_WINDOW'] = 'offscreen'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)    # offscreen: no requestProperties
import main as M
from panda3d.core import Filename
def shot(name):
    for _ in range(3): M.app.step()
    ok = ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(os.environ.get('SHOT_DIR', r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports'), name))); print('shot', name, ok, flush=True)
P = M.scene_setup.player; P.cursor.enabled = False    # pink diamond = FirstPersonController crosshair (camera.ui, scale .008), huge in an offscreen buffer
P.position = (0, 6, -9); P.camera_pivot.rotation_x = 32; P.rotation_y = 0
from src.algo import growN as _g; print('ZSOFT in window process =', _g.ZSOFT, 'LA =', _g.ZSOFTLA)
M.stepper.key('C'); M.growview.draw(M.stepper.snapshot); shot('vsurf_smoke_off.png')
M.growview.toggle_vsurf(); print(M.growview.vlabel()); shot('vsurf_smoke_on.png')
M.growview.toggle_layer(1); shot('vsurf_smoke_layer2_hidden.png'); M.growview.toggle_layer(1)
P.position = (0.5, 3.2, -3.6); P.camera_pivot.rotation_x = 40; shot('vsurf_smoke_close.png')
M.growview.toggle_vsurf(); M.growview.toggle_common(); print(M.growview.vlabel()); shot('vsurf_smoke_common.png'); P.position = (0, 6, -9); P.camera_pivot.rotation_x = 32; M.growview.toggle_vsurf(); shot('vsurf_smoke_both.png'); M.growview.toggle_vsurf(); M.growview.toggle_common(); print('off ok', M.growview.vlabel() == ''); print('SMOKE OK')
