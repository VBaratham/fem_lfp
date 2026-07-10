"""Shared polyline-geometry helpers."""
import numpy as np

from fem_lfp._geom import arc_fraction_of_projection, point_at_arc_fraction


def test_point_at_arc_fraction_endpoints_and_midpoint():
    # An L-shaped polyline of two equal-length legs (total arc = 2).
    poly = np.array([[0.0, 0, 0], [1.0, 0, 0], [1.0, 1.0, 0]])
    assert np.allclose(point_at_arc_fraction(poly, 0.0), [0, 0, 0])
    assert np.allclose(point_at_arc_fraction(poly, 1.0), [1, 1, 0])
    # Half the arc length lands exactly on the corner.
    assert np.allclose(point_at_arc_fraction(poly, 0.5), [1, 0, 0])


def test_point_at_arc_fraction_clamps():
    poly = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    assert np.allclose(point_at_arc_fraction(poly, -1.0), [0, 0, 0])
    assert np.allclose(point_at_arc_fraction(poly, 2.0), [10, 0, 0])


def test_arc_fraction_of_projection():
    poly = np.array([[0.0, 0, 0], [10.0, 0, 0]])
    # A point above x=2.5 projects to arc fraction 0.25.
    assert np.isclose(arc_fraction_of_projection(np.array([2.5, 5.0, 0]), poly),
                      0.25)
    # Beyond the far end clamps to 1.0.
    assert np.isclose(arc_fraction_of_projection(np.array([99.0, 1.0, 0]), poly),
                      1.0)
