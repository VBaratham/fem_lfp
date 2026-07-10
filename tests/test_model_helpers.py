"""ExtracellularModel helpers that don't need the FEM/NEURON stack."""
import numpy as np
import pytest

from fem_lfp import MESHERS, SectionGeometry
from fem_lfp.model import _as_probe_array, _looks_like_z_cylinder


def test_probe_array_accepts_single_point():
    out = _as_probe_array([1.0, 2.0, 3.0])
    assert out.shape == (1, 3)


def test_probe_array_accepts_list_of_points():
    out = _as_probe_array([(1.0, 0, 0), (2.0, 0, 0)])
    assert out.shape == (2, 3)


def test_probe_array_rejects_bad_shape():
    with pytest.raises(ValueError):
        _as_probe_array([[1.0, 2.0], [3.0, 4.0]])   # (2, 2), not (P, 3)


def test_z_cylinder_detection():
    # Centered, z-aligned straight cable → cylinder-shaped.
    g = SectionGeometry(
        name="cyl",
        points_um=np.array([[0.0, 0.0, -100.0], [0.0, 0.0, 100.0]]),
        diameters_um=np.array([5.0, 5.0]),
        nseg=41,
    )
    assert _looks_like_z_cylinder(g)


def test_offaxis_not_z_cylinder():
    # Same length but offset off the z axis → not the cylinder mesher's case.
    g = SectionGeometry(
        name="tilted",
        points_um=np.array([[0.0, 0.0, 0.0], [80.0, 60.0, 100.0]]),
        diameters_um=np.array([5.0, 5.0]),
        nseg=10,
    )
    assert not _looks_like_z_cylinder(g)


def test_meshers_catalogue():
    assert {"cylinder", "branched", "body_fitted"} <= set(MESHERS)
