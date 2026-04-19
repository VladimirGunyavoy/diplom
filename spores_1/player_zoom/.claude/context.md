# Player Zoom - Project Context

## Overview
Player Zoom is a **simplified sandbox** extracted from the main v16_picker project for independent development of camera, zoom, and control mechanics.

**Location:** `c:\GitHub\diplom\spores\player_zoom\`
**Original project:** `c:\GitHub\diplom\spores\v16_picker\`

## Current State

### What Works ✅
- FirstPersonController with extended controls
- Zoom system with **invariant point** (look point calculation)
- Grid floor (40x40) with coordinate axes
- Window/monitor management
- Clean, English-only codebase
- Auto-restart watcher

### Key Features
- **Zoom around look point**: Q/E keys zoom while maintaining the point camera is looking at
- **Invariant point calculation**: `identify_invariant_point()` calculates where camera ray intersects ground (y=0)
- **Scalable objects**: All objects registered in ZoomManager are transformed uniformly

### Controls
```
MOVEMENT: WASD, Space/Shift, Mouse
CURSOR: Alt (lock/unlock)
EXIT: Escape
ZOOM: Q (out), E (in), R (reset)
FULLSCREEN: F11
DEBUG: H (debug info)
```

## Project Structure

```
player_zoom/
├── main.py                    # Entry point with 3 test spheres
├── run.py                     # Launcher with watcher
├── assets/
│   └── arrow.obj              # 3D model for axes
├── src/
│   ├── scalable.py            # Scalable, ScalableFloor (33 lines)
│   ├── frame.py               # Coordinate axes (X,Y,Z)
│   ├── scene_setup.py         # Player + camera (107 lines)
│   ├── zoom_manager.py        # Zoom with invariant point (105 lines)
│   ├── window_manager.py      # Window/monitor management
│   ├── color_manager.py       # Color configuration
│   ├── input_manager.py       # Input handling
│   └── watcher.py             # Auto-restart on exit
└── config/
    └── colors.json            # (if exists)
```

## What Was Cleaned Up 🧹

### Removed from original v16_picker:
- ❌ `visual_manager` - ghost visualization
- ❌ Ghost detection logic throughout zoom_manager
- ❌ `ui_manager`, `ui_constants` - unused UI system
- ❌ `spores_scale` - separate scaling for different object types
- ❌ `look_point_subscribers` - unused subscription system
- ❌ Debug methods: `print_all_objects()`, `show_all_objects_with_ghosts()`, etc.
- ❌ ID generators: `get_unique_spore_id()`, `get_unique_link_id()`
- ❌ Unused classes: `ScalableFrame`, `Link`
- ❌ All Russian text and comments
- ❌ Commented-out code blocks

### Simplified code:
- **zoom_manager.py**: 446 lines → 105 lines (↓76%)
- **scalable.py**: 45 lines → 33 lines (↓27%)
- Clean English-only codebase
- No ghost logic, no UI complexity

## Key Implementation Details

### ZoomManager
```python
class ZoomManager:
    def __init__(self, scene_setup, color_manager=None):
        self.zoom_fact = 1 + 1/8
        self.a_transformation = 1.0  # Scaling coefficient
        self.b_translation = np.array([0, 0, 0])  # Translation vector
        self.objects = {}  # Registered objects

    def identify_invariant_point(self) -> Tuple[float, float]:
        """Calculate where camera ray intersects ground (y=0)"""
        # Uses camera position, rotation to calculate look point

    def change_zoom(self, sign: int):
        """Zoom while maintaining invariant point"""
        inv = self.identify_invariant_point()
        inv_3d = np.array([inv[0], 0, inv[1]])

        zoom_multiplier = self.zoom_fact ** sign
        self.a_transformation *= zoom_multiplier
        self.b_translation = zoom_multiplier * self.b_translation + (1 - zoom_multiplier) * inv_3d
        self.update_transform()
```

### Scalable Objects
```python
class Scalable(Entity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.real_position = np.array(self.position)
        self.real_scale = np.array(self.scale)

    def apply_transform(self, a: float, b: np.ndarray, **kwargs):
        """Apply zoom transformation"""
        self.position = self.real_position * a + b
        self.scale = self.real_scale * a
```

## Original Project Reference

The original v16_picker has more complex features that might need to be ported:
- **Spore system**: Nodes with pendulum physics
- **Link system**: Connections between spores
- **Tree optimization**: Area minimization algorithms
- **Valence system**: Route alternation logic
- **Graph visualization**: Real-time graph updates

## Common Tasks

### Adding new object types:
1. Create class inheriting from `Scalable`
2. Register in `zoom_manager.register_object(obj, name='...')`
3. Object will automatically respond to zoom

### Modifying zoom behavior:
- Adjust `zoom_fact` in ZoomManager.__init__
- Modify `identify_invariant_point()` for different look point calculation

### Testing:
```bash
python .\player_zoom\run.py      # with auto-restart
python .\player_zoom\main.py     # direct
```

## Next Steps / TODO

The user wants to port more features from v16_picker. Likely candidates:
- Spore objects (if simplified)
- Specific visualization components
- Additional input/control mechanics
- Performance optimizations from the original

## Important Notes

- **All code must be in English** - no Russian text
- **Keep it simple** - this is a sandbox, not the full v16_picker
- **No ghost logic** - we removed that complexity
- **Uniform scaling** - all objects scale together (no separate spores_scale)
- **Clean imports** - removed unused UI managers, etc.

## Development Environment
- Platform: Windows 11 Pro
- Shell: bash (Unix syntax)
- Python: 3.14.0
- Ursina: 8.3.0
- IDE: VSCode with Claude Code extension

---

**Ready to continue development!** 🚀
