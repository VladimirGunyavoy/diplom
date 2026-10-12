import os, sys
os.environ['V8_WINDOW'] = 'offscreen'; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main as M
from panda3d.core import Filename
def shot(name):
    for _ in range(3): M.app.step()
    print('shot', name, ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(os.environ.get('SHOT_DIR', r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports'), name))), flush=True)
P = M.scene_setup.player; P.cursor.enabled = False; P.position = (0.5, 3.2, -3.6); P.camera_pivot.rotation_x = 40; P.rotation_y = 0
M.stepper.key('C'); G = M.growview; G.draw(M.stepper.snapshot); G.toggle_common()
for i in range(3): print(G.vlabel()); shot('vcalc_%d_%s.png' % (i, G.vcalc)); G.cycle_vcalc()
print(G.vlabel()); G.toggle_common(); print('off', G.vlabel() == ''); print('SMOKE OK')
