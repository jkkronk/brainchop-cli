"""How a model argument is read: a built-in model name or a custom model directory."""

from pathlib import Path

import pytest

from brainchop import api


@pytest.fixture
def cwd(home, tmp_path, monkeypatch):
    """Work in tmp_path, which holds an empty directory named like a built-in model."""
    (tmp_path / "tissue_fast").mkdir()
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_registered_name_means_builtin_despite_same_named_directory(cwd):
    assert api._custom_model_dir("tissue_fast") is None


@pytest.mark.parametrize(
    "spelling", ["./tissue_fast", "{dir}", "file://{dir}", "file://tissue_fast"]
)
def test_directory_written_as_path_is_custom(cwd, spelling):
    model_dir = cwd / "tissue_fast"
    found = api._custom_model_dir(spelling.format(dir=model_dir))
    assert found is not None and found.samefile(model_dir)


def test_file_url_expands_home(cwd, home):
    (home / "models" / "mine").mkdir(parents=True)
    found = api._custom_model_dir("file://~/models/mine")
    assert found is not None and found.samefile(home / "models" / "mine")


def test_unregistered_directory_name_is_custom(cwd):
    (cwd / "my_model").mkdir()
    assert api._custom_model_dir("my_model") == Path("my_model")


def test_missing_directory_is_looked_up_by_name(cwd):
    assert api._custom_model_dir("no_such_model") is None


def test_load_model_uses_builtin_despite_same_named_directory(cwd, monkeypatch):
    from brainchop import utils  # importing it reads ~/.cache/brainchop/models.json

    class BuiltIn(Exception):
        pass

    def find_pth_files(model_name):
        raise BuiltIn(model_name)

    monkeypatch.setattr(utils, "find_pth_files", find_pth_files)
    with pytest.raises(BuiltIn):
        api._load_model("tissue_fast")


def test_cache_name():
    assert api._model_cache_name("tissue_fast") == "tissue_fast"
    assert api._model_cache_name("/models/my_model") == "my_model"
    assert api._model_cache_name("file:///models/my_model") == "my_model"
