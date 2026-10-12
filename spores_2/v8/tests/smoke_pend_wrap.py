import os, sys
os.environ["V8_WINDOW"] = "offscreen"; os.chdir(r'C:\GitHub\claude-work\projects\spore\spores_2\v8'); sys.path.insert(0, os.getcwd())
import ursina; _M = type(ursina.mouse); _M.locked = property(lambda s: False, lambda s, v: None); _M.visible = property(lambda s: True, lambda s, v: None)
import main_pend as M
from panda3d.core import Filename
G = M.growview; M.stepper.key('C'); G.draw(M.stepper.snapshot)
for _ in range(5): M.app.step(); G.poll()
P = M.scene_setup.player; P.cursor.enabled = False; P.position = (0, 16, -8); P.camera_pivot.rotation_x = 55
for _ in range(3): M.app.step()
print('shot', ursina.application.base.win.save_screenshot(Filename.from_os_specific(os.path.join(os.environ.get('SHOT_DIR', r'C:\GitHub\claude-work\projects\spore\spores_2\v8\reports'), 'pend_wrap.png'))))
print('WRAP DONE')
