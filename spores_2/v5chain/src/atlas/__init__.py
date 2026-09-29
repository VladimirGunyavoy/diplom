"""Атлас спор для двойного интегратора (АТЛАС docs/spore_atlas_double_integrator.md). numpy, без Ursina."""
from .coords import A, c_plus, c_minus
from .lattice import key, build_lattice, build_edges
from .solve import cost_to_go
from .interp import interp_T
from .reference import T_star
