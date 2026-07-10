"""Line-source approximation: analytical sanity checks (numpy-only)."""
import numpy as np

from fem_lfp import line_source_v_e


def test_monopole_far_field_limit():
    """A short segment seen from far away approaches a point source:
    V ≈ I / (4 π σ r)."""
    sigma = 0.3
    I_nA = 1.0
    L_um = 1.0
    r_um = 1000.0  # r / L = 1000 → deep in the monopole regime

    p1 = np.array([[0.0, 0.0, -L_um / 2]])
    p2 = np.array([[0.0, 0.0, +L_um / 2]])
    imem = np.array([[I_nA]])                 # (S=1, T=1)
    probe = np.array([[r_um, 0.0, 0.0]])      # perpendicular, far

    v = line_source_v_e(probe, p1, p2, imem, sigma_S_per_m=sigma)  # (P,T) volts
    monopole = (I_nA * 1e-9) / (4 * np.pi * sigma * (r_um * 1e-6))
    assert np.isclose(v[0, 0], monopole, rtol=1e-3)


def test_superposition():
    """V_e is linear in the per-segment currents."""
    p1 = np.array([[0.0, 0, -1.0], [10.0, 0, -1.0]])
    p2 = np.array([[0.0, 0, 1.0], [10.0, 0, 1.0]])
    probe = np.array([[50.0, 0.0, 0.0]])

    a = line_source_v_e(probe, p1, p2, np.array([[2.0], [0.0]]))
    b = line_source_v_e(probe, p1, p2, np.array([[0.0], [3.0]]))
    both = line_source_v_e(probe, p1, p2, np.array([[2.0], [3.0]]))
    assert np.allclose(a + b, both)


def test_sign_follows_current():
    """Outward-positive current gives positive V_e nearby; flipping the
    current flips the sign."""
    p1 = np.array([[0.0, 0.0, -1.0]])
    p2 = np.array([[0.0, 0.0, 1.0]])
    probe = np.array([[20.0, 0.0, 0.0]])
    pos = line_source_v_e(probe, p1, p2, np.array([[1.0]]))
    neg = line_source_v_e(probe, p1, p2, np.array([[-1.0]]))
    assert pos[0, 0] > 0
    assert np.isclose(pos[0, 0], -neg[0, 0])
