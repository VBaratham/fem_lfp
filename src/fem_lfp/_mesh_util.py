"""dolfinx-only mesh helpers shared by the branched and body-fitted meshers.

Both meshers build a full EMI-style mesh (ICS + ECS), extract the ECS as a
submesh, and must carry the parent's facet tags onto that submesh. This is
the one implementation of that transfer. No fem_neuron dependency here.
"""
from __future__ import annotations

import numpy as np

import dolfinx


def make_parent_to_sub_facet_locator(parent, sub):
    """Return ``locate(parent_facet) -> sub_facet_index | None``.

    A membrane facet that was *interior* in the parent EMI mesh becomes a
    *boundary* facet of the extracted ECS submesh, so only the submesh's
    exterior facets can match a tagged parent facet — we key just those (a
    big speedup vs. keying every facet). Facets are matched by their three
    vertex coordinates, rounded to 1e-9 of the mesh unit and sorted so
    orientation doesn't matter. The caller must have created the submesh's
    facet→cell connectivity already (needed for exterior_facet_indices).
    """
    fdim = parent.topology.dim - 1
    parent.topology.create_connectivity(fdim, 0)
    sub.topology.create_connectivity(fdim, 0)
    p_f_to_v = parent.topology.connectivity(fdim, 0)
    s_f_to_v = sub.topology.connectivity(fdim, 0)
    parent_x = parent.geometry.x
    sub_x = sub.geometry.x

    def _key(coords3: np.ndarray) -> tuple:
        cq = np.round(coords3, 9)
        order = np.lexsort(cq.T[::-1])
        return tuple(cq[order].flatten())

    sub_key_to_idx: dict[tuple, int] = {}
    for sf in dolfinx.mesh.exterior_facet_indices(sub.topology):
        sub_key_to_idx[_key(sub_x[s_f_to_v.links(sf)])] = int(sf)

    def locate(pf: int):
        return sub_key_to_idx.get(_key(parent_x[p_f_to_v.links(pf)]))

    return locate
