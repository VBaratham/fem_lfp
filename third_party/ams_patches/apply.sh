#!/usr/bin/env bash
# Apply fem_lfp's AMS patches to a local clone of Alpha_Mesh_Swc.
#
# Usage:  apply.sh /path/to/Alpha_Mesh_Swc
#
# The patch files are unified diffs targeting the AMS clone root
# (-p1). After running, AMS is in the patched state; remove via
# ``git -C $1 reset --hard`` if you want to revert.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "usage: $0 /path/to/Alpha_Mesh_Swc"
    exit 2
fi

ams_root="$1"
patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$ams_root" ] || [ ! -f "$ams_root/mesh_swc.py" ]; then
    echo "error: $ams_root doesn't look like an AMS clone (no mesh_swc.py)" >&2
    exit 1
fi

for p in "$patch_dir"/*.patch; do
    echo "applying $(basename "$p")"
    patch -p1 -d "$ams_root" --no-backup-if-mismatch < "$p"
done

echo
echo "done. set FEM_NEURON_AMS_ROOT=$ams_root to use this patched clone."
