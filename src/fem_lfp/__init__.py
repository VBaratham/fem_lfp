"""fem_lfp — hybrid NEURON + ECS-only 3D FEM for LFP forward modeling.

Public interface::

    from fem_lfp import ExtracellularModel

    model = ExtracellularModel(h.allsec(), probes_um)   # before finitialize
    h.finitialize(-65); h.continuerun(30)
    result = model.solve()
    result.plot("lfp.png")

See :class:`ExtracellularModel` for the knobs (mesher, conductivity, mesh
sizing). Everything else in the package is machinery the model drives.

Progress/diagnostic output goes through the standard ``logging`` module
(logger name ``fem_lfp``); call ``logging.basicConfig(level=logging.INFO)``
to see it. The library attaches a NullHandler so it stays silent by default.
"""
import logging as _logging

__version__ = "0.0.1"

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from .lsa import line_source_v_e
from .model import MESHERS, ExtracellularModel, ExtracellularResult
from .neuron_sim import SectionGeometry

__all__ = [
    "ExtracellularModel",
    "ExtracellularResult",
    "SectionGeometry",
    "MESHERS",
    "line_source_v_e",
    "__version__",
]
