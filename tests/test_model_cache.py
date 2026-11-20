"""Downloads into the ~/.cache/brainchop model cache, with requests.get faked out."""

import errno
import json
from pathlib import Path

import pytest
import requests

URL = "https://example.invalid/model.pth"
PTH = bytes(range(256)) * 100
BIN = b"tfjs weights"
CONFIG = b'{"layers": []}'
LISTING = {"fresh": {"folder": "fresh", "description": "listing served by the fake server"}}


class FakeResponse:
    """Just enough of requests.Response: a status code and a body that can break off."""

    def __init__(self, status=200, body=b"", broken=False):
        self.status_code, self.body, self.broken = status, body, broken

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Error", response=self)

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self.body), chunk_size):
            yield self.body[i : i + chunk_size]
        if self.broken:
            raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")

    def json(self):
        return json.loads(self.body)


class FakeServer:
    """Stands in for requests.get: answers by file name (404 if unknown), logging each name."""

    def __init__(self):
        self.files = {}
        self.requested = []

    def get(self, url, **kwargs):
        name = url.rsplit("/", 1)[-1]
        self.requested.append(name)
        response = self.files.get(name, FakeResponse(404))
        if isinstance(response, Exception):
            raise response
        return response


@pytest.fixture
def utils(home, monkeypatch):
    """brainchop.utils, imported only once HOME points at the temporary cache."""
    from brainchop import utils

    # In case another test imported brainchop.utils first, with a different listing.
    listing = json.loads((home / ".cache" / "brainchop" / "models.json").read_text())
    monkeypatch.setattr(utils, "AVAILABLE_MODELS", listing)
    return utils


@pytest.fixture
def server(utils, monkeypatch):
    """A FakeServer in place of requests.get."""
    fake = FakeServer()
    monkeypatch.setattr(utils.requests, "get", fake.get)
    return fake


def cached(home, utils, model):
    """Names of the files in a model's cache folder."""
    folder = home / ".cache" / "brainchop" / "models" / utils.AVAILABLE_MODELS[model]["folder"]
    return sorted(p.name for p in folder.iterdir())


@pytest.mark.parametrize("as_type", [str, Path])
def test_download_writes_exact_bytes(utils, server, tmp_path, as_type):
    server.files["model.pth"] = FakeResponse(body=PTH)
    dest = tmp_path / "cache" / "model.pth"
    utils.download(URL, as_type(dest))
    assert dest.read_bytes() == PTH
    assert [p.name for p in dest.parent.iterdir()] == ["model.pth"]


@pytest.mark.parametrize("as_type", [str, Path])
def test_interrupted_download_leaves_no_file(utils, server, tmp_path, as_type):
    server.files["model.pth"] = FakeResponse(body=PTH[:1000], broken=True)
    dest = tmp_path / "cache" / "model.pth"
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        utils.download(URL, as_type(dest))
    assert list(dest.parent.iterdir()) == []


@pytest.mark.parametrize(
    "failure",
    [
        FakeResponse(503),
        FakeResponse(body=PTH[:1000], broken=True),
        requests.ConnectionError("Connection refused"),
    ],
    ids=["503", "truncated", "connection-error"],
)
def test_failed_pth_download_is_retried_next_run(utils, server, home, failure):
    server.files.update(
        {
            "model.json": FakeResponse(body=CONFIG),
            "model.pth": failure,
            "model.bin": FakeResponse(body=BIN),
        }
    )
    with pytest.raises(requests.RequestException):
        utils.find_pth_files("tissue_fast")
    assert "model.bin" not in server.requested
    assert cached(home, utils, "tissue_fast") == ["model.json"]

    server.files["model.pth"] = FakeResponse(body=PTH)
    json_fn, weights_fn = utils.find_pth_files("tissue_fast")
    assert json_fn.read_bytes() == CONFIG
    assert weights_fn.name == "model.pth" and weights_fn.read_bytes() == PTH
    assert server.requested == ["model.json", "model.pth", "model.pth"]


def test_missing_pth_falls_back_to_bin(utils, server, home):
    server.files.update(
        {
            "model.json": FakeResponse(body=CONFIG),
            "model.pth": FakeResponse(404),
            "model.bin": FakeResponse(body=BIN),
        }
    )
    json_fn, weights_fn = utils.find_pth_files("tissue_fast")
    assert weights_fn.name == "model.bin" and weights_fn.read_bytes() == BIN
    assert server.requested == ["model.json", "model.pth", "model.bin"]
    assert cached(home, utils, "tissue_fast") == ["model.bin", "model.json"]

    # From then on the cached model.bin is used without asking the server again.
    assert utils.find_pth_files("tissue_fast") == (json_fn, weights_fn)
    assert len(server.requested) == 3


def test_interrupted_sae_download_is_retried_next_run(utils, server, home):
    server.files["model.pth"] = FakeResponse(body=PTH[:1000], broken=True)
    with pytest.raises(requests.exceptions.ChunkedEncodingError):
        utils.find_sae_files("robust_tissue")
    assert cached(home, utils, "robust_tissue") == []

    server.files["model.pth"] = FakeResponse(body=PTH)
    assert utils.find_sae_files("robust_tissue").read_bytes() == PTH


@pytest.mark.parametrize(
    "content",
    [b'{\n  "tissue_fast": {\n    "folder": "model5_gw_ae",\n', b"", b"\xff\xfe\x00"],
    ids=["truncated", "empty", "not-utf8"],
)
def test_corrupt_listing_is_downloaded_again(utils, server, home, content):
    listing = home / ".cache" / "brainchop" / "models.json"
    listing.write_bytes(content)
    server.files["models.json"] = FakeResponse(body=json.dumps(LISTING).encode())
    assert utils.load_models() == LISTING
    assert json.loads(listing.read_text()) == LISTING


def test_valid_listing_is_read_without_a_request(utils, server, home):
    listing = home / ".cache" / "brainchop" / "models.json"
    assert utils.load_models() == json.loads(listing.read_text())
    assert server.requested == []


def test_failed_refresh_keeps_listing(utils, server, home):
    listing = home / ".cache" / "brainchop" / "models.json"
    before = listing.read_bytes()
    server.files["models.json"] = FakeResponse(503)
    with pytest.raises(requests.HTTPError):
        utils.update_models()
    assert listing.read_bytes() == before


def test_interrupted_listing_write_keeps_listing(utils, server, home, monkeypatch):
    listing = home / ".cache" / "brainchop" / "models.json"
    before = listing.read_bytes()
    server.files["models.json"] = FakeResponse(body=json.dumps(LISTING).encode())

    def dump_until_disk_full(obj, f, **kwargs):
        f.write(json.dumps(obj, **kwargs)[:10])
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(utils.json, "dump", dump_until_disk_full)
    with pytest.raises(OSError):
        utils.update_models()
    assert listing.read_bytes() == before
