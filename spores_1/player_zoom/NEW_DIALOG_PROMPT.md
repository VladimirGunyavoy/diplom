# Player Zoom - Continuation Prompt

Use this to start a new dialogue for continuing work on Player Zoom:

---

## Quick Context

I'm working on **Player Zoom** - a simplified camera/zoom sandbox extracted from a larger project (v16_picker).

**Project location:** `c:\GitHub\diplom\spores\player_zoom\`
**Original project:** `c:\GitHub\diplom\spores\v16_picker\`

### Current State
- ✅ Working zoom system with invariant point (look point)
- ✅ Clean codebase (English only, no ghost logic, simplified)
- ✅ Basic controls: Q/E zoom, R reset, Alt cursor toggle
- ✅ Grid floor + coordinate axes + 3 test spheres

### What I Need
I want to port **[DESCRIBE WHAT YOU WANT TO PORT]** from the original v16_picker project to player_zoom.

### Key Files to Know About
- `main.py` - entry point with 3 test spheres
- `src/zoom_manager.py` - zoom logic (105 lines, cleaned up)
- `src/scalable.py` - base classes for zoomable objects (33 lines)
- `src/scene_setup.py` - player and camera setup

### Detailed Context
Full context is in `.claude/context.md` - read it first if you need complete details about what was cleaned up and how the code works.

### Rules
- All code must be in **English** (no Russian)
- Keep it **simple** - this is a sandbox, not the full project
- No ghost logic or complex UI - we removed that
- All objects scale uniformly (no separate spores_scale)

---

## Example Usage

**For porting specific feature:**
```
I want to port the [feature name] from v16_picker to player_zoom.

Original location: v16_picker/src/[path]/[file].py

What it does: [brief description]

What I need in player_zoom: [simplified version]
```

**For debugging/fixing:**
```
I have an issue with [feature] in player_zoom.

Current behavior: [what happens]
Expected behavior: [what should happen]

Relevant files: [list files]
```

---

Ready to continue! 🚀
