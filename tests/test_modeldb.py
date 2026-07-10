"""ModelDB fetch/compile helpers that don't touch the network or NEURON."""
from fem_lfp import modeldb


def test_download_url_format():
    assert modeldb.MODELDB_DOWNLOAD.format(id=139653) == (
        "https://modeldb.science/download/139653"
    )


def test_arch_built_detection(tmp_path):
    # Nothing compiled yet.
    assert not modeldb._arch_is_built(tmp_path)
    # Simulate nrnivmodl output: <arch>/libnrnmech.dylib
    arch = tmp_path / "arm64"
    arch.mkdir()
    (arch / "libnrnmech.dylib").write_bytes(b"")
    assert modeldb._arch_is_built(tmp_path)
