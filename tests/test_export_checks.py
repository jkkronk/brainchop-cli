"""export()'s free-memory guard and its force=True override, without running the exporter."""

import os
import sys
from types import ModuleType

import pytest

from brainchop import api


class Stop(Exception):
    pass


@pytest.fixture
def low_free_memory(tmp_path, monkeypatch):
    """REQUIRE_FREE_GB=1 and a vm_stat, first on PATH, that reports 100 free pages."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    vm_stat = bin_dir / "vm_stat"
    vm_stat.write_text('#!/bin/sh\necho "Pages free:                               100."\n')
    vm_stat.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("REQUIRE_FREE_GB", "1")
    # Stands in for brainchop.export_model, which needs a real export backend.
    exporter = ModuleType("brainchop.export_model")
    exporter.export_model = None
    monkeypatch.setitem(sys.modules, "brainchop.export_model", exporter)

    def load_model(*args, **kwargs):
        raise Stop  # export() got past its checks

    monkeypatch.setattr(api, "_load_model", load_model)


def test_force_skips_free_memory_check(low_free_memory, tmp_path):
    with pytest.raises(Stop):
        api.export("tissue_fast", str(tmp_path / "export"), force=True)


def test_free_memory_check_blocks_export_without_force(low_free_memory, tmp_path):
    with pytest.raises(RuntimeError, match="free memory, need 1GB"):
        api.export("tissue_fast", str(tmp_path / "export"))
