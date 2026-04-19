from ursina import *
import numpy as np
from .spore import Spore

class ZoomManager:
    def __init__(self, scene_setup):
        self.zoom_x = 1
        self.zoom_y = 1
        self.zoom_z = 1

        self.zoom_fact = 1 + 1/16
        self.spores_scale = 1

        self.txt = Text(text='', position=(0, 0, 0))

        self.objects = {}

        self.scene_setup = scene_setup
        self.instructions = Text(
            text=dedent('''
            <white>ZOOM:
            E - zoom in
            T - zoom out
            R - reset
            1 - larger 
                spores
            2 - smaller 
                spores
            ''').strip(),
            position=(-0.75, 0.2),
            scale=0.75,
            color=color.white,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7),
        )

        
        self.look_point_text = Text(
            text='LOOK POINT:\nX: 0.000\nZ: 0.000', 
            position=(0.5, 0.45, 0), 
            scale=0.75, 
            color=color.white,
            background=True,
            background_color=color.rgba(0, 0, 0, 0.7),
            origin=(-0.5, 0.5),
            parent=camera.ui,
            font='VeraMono.ttf',
            font_size=16,
        )


    def register_object(self, object, name=None):
        if name is None:
            name = f"obj_{len(self.objects)}"
        self.objects[name] = object


    def identify_invariant_point(self):

        player = self.scene_setup.player

        # self.txt.text = f'pos: {player.position}'
        # self.txt.text += f'\ncpos: {player.camera_pivot.world_position}'
        # self.txt.text += f'\npsi: {round(player.rotation_y, 3)}'
        # self.txt.text += f'\nphi: {round(player.camera_pivot.rotation_x, 3)}'

        psi = np.radians(self.scene_setup.player.rotation_y)
        phi = np.radians(self.scene_setup.player.camera_pivot.rotation_x)

        h = self.scene_setup.player.camera_pivot.world_position.y
        d = (h / np.tan(phi)) 

        # self.txt.text += f'\nh: {round(h, 3)}'
        # self.txt.text += f'\nd: {round(d, 3)}'

        dx = d * np.sin(psi)
        dy = d * np.cos(psi) 

        x_0 = self.scene_setup.player.camera_pivot.world_position.x + dx
        z_0 = self.scene_setup.player.camera_pivot.world_position.z + dy


        look_point_coords = np.array([x_0, z_0])

        frame_origin_coords = self.scene_setup.frame.origin.world_position
        frame_origin_coords = np.array([frame_origin_coords[0], frame_origin_coords[2]])

        frame_origin_coords = self.scene_setup.frame.origin.world_position
        frame_origin_coords = np.array([frame_origin_coords[0], frame_origin_coords[2]])
        current_scale = self.scene_setup.frame.x_axis.scale[0]

        scaled_look_point_coords = (look_point_coords - frame_origin_coords) / current_scale

        x_1 = scaled_look_point_coords[0]
        z_1 = scaled_look_point_coords[1]

        self.look_point_text.text = 'LOOK POINT:\n'
        self.look_point_text.text += f'({x_1:6.3f}, {z_1:6.3f})\n'
        self.look_point_text.text += f'common scale: {current_scale:6.3f}\n'
        self.look_point_text.text += f'spores scale: {self.spores_scale:6.3f}'
        self.look_point_text.background = True

        return x_0, z_0
    
    def zoom_obj(self, obj, sign):
        pos_ = np.array([obj.position[0], obj.position[2]])

        inv = self.identify_invariant_point()

        new_pos = (pos_ - inv) * self.zoom_fact ** sign + inv
        new_pos_vec = [new_pos[0], obj.position[1], new_pos[1]]

        scale = obj.scale * self.zoom_fact ** sign

        obj.position = new_pos_vec
        obj.scale = scale

    
    def reset_obj(self, obj):
        obj.position = obj.original_position
        obj.scale = obj.original_scale
    
    def zoom_all(self, sign):
        for name, obj in self.objects.items():
            self.zoom_obj(obj, sign)
        self.scene_setup.player.speed *= (self.zoom_fact ** sign - 1)/2 + 1
        # self.scene_setup.player.y += 0.2 * self.zoom_fact ** sign
        self.spores_scale *= self.zoom_fact ** sign

    def reset_all(self):
        for name, obj in self.objects.items():
            self.reset_obj(obj)
        self.scene_setup.player.speed = self.scene_setup.base_speed
        self.scene_setup.player.position = self.scene_setup.base_position
        self.spores_scale = 1


    def scale_spores(self, sign):
        # print('hello from scale_spores')
        
        self.spores_scale *= self.zoom_fact ** sign
        for name, obj in self.objects.items():
            if isinstance(obj, Spore):
                obj.scale = obj.scale * self.zoom_fact ** sign
                # print(name, obj)
            # print('end of scale_spores')
            # print()


    def input_handler(self, key):
        if key == 'e':
            self.zoom_all(1)
        elif key == 't':
            self.zoom_all(-1)
        elif key == 'r':
            self.reset_all()
        elif key == '1':
            self.scale_spores(1)
        elif key == '2':
            self.scale_spores(-1)

        



