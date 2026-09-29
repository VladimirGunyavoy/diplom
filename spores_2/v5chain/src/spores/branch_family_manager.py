"""
BranchFamilyManager - Factory for BranchFamily objects
=======================================================

Creates BranchFamily instances and registers them as tickables automatically.
"""

from typing import Dict, TYPE_CHECKING
from .boundary_ray_family import BranchFamily

if TYPE_CHECKING:
    from ..core.object_manager import ObjectManager
    from ..core.shared_context import SharedContext
    from .spore import GhostSpore


class BranchFamilyManager:

    def __init__(self, object_manager: "ObjectManager"):
        self._object_manager = object_manager
        self._families: Dict[str, BranchFamily] = {}

    def create(self, name: str, root: "GhostSpore", template: tuple,
               ctx: "SharedContext", color_key: str = 'ray',
               color_suffix: str = 'plus_u') -> BranchFamily:
        family = BranchFamily(root, template=template, ctx=ctx,
                              color_key=color_key, color_suffix=color_suffix, name=name)
        self._object_manager.register_tickable(family)
        self._families[name] = family
        return family

    def get(self, name: str) -> BranchFamily:
        return self._families[name]
