"""Root pytest configuration for Liposome-Aggregate-Analyse.

Provides
--------
- ``--data-dir`` CLI option (or ``LAA_TEST_DATA_DIR`` env-var) to override
  the directory containing leaflet pickle fixtures.  Defaults to
  ``test/example/analyzer/``.
- ``--run-slow`` CLI flag to include tests decorated with ``@pytest.mark.slow``.
- Auto-skip logic for ``slow`` and ``gpu`` markers.
- Session-scoped fixtures for data paths used by the analyzer tests.

Examples
--------
Run tests with local fixture data (the default)::

    uv run pytest

Run tests with server data located elsewhere::

    uv run pytest --data-dir /share/home/qjxu/outputs

Or via the environment variable::

    LAA_TEST_DATA_DIR=/share/home/qjxu/outputs uv run pytest

Include slow tests::

    uv run pytest --run-slow
"""

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "test" / "example" / "analyzer"


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--data-dir",
        action="store",
        default=os.environ.get("LAA_TEST_DATA_DIR", None),
        help=(
            "Directory containing leaflet pickle fixtures "
            "(vesicle_leaflet.pickle, vesicle_leaflet_args.pickle, "
            "vesicle_data.pickle).  "
            "Env-var: LAA_TEST_DATA_DIR.  "
            "Default: test/example/analyzer/"
        ),
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as @pytest.mark.slow.",
    )


# ---------------------------------------------------------------------------
# Collection hooks — auto-skip
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on markers and runtime environment."""

    # --- slow ---
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    # --- gpu ---
    gpu_available = _check_gpu()
    if not gpu_available:
        skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def _check_gpu():
    """Return True if a CUDA device is reachable."""
    try:
        import cupy

        cupy.cuda.Device(0).compute_capability
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Session fixtures — data paths
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def project_root():
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def data_dir(request):
    """Resolve the directory that holds the leaflet pickle fixtures.

    Resolution order:
        1. ``--data-dir`` CLI option
        2. ``LAA_TEST_DATA_DIR`` environment variable
        3. Default: ``test/example/analyzer/``
    """
    custom = request.config.getoption("--data-dir")
    if custom is not None:
        d = Path(custom)
        if not d.exists():
            pytest.fail("Specified data directory does not exist: {}".format(d))
        return d
    return DEFAULT_DATA_DIR


@pytest.fixture(scope="session")
def leaflet_pickle(data_dir):
    """Path to ``vesicle_leaflet.pickle``."""
    p = data_dir / "vesicle_leaflet.pickle"
    if not p.exists():
        pytest.skip("Leaflet pickle not found: {}".format(p))
    return str(p)


@pytest.fixture(scope="session")
def args_pickle(data_dir):
    """Path to ``vesicle_leaflet_args.pickle``."""
    p = data_dir / "vesicle_leaflet_args.pickle"
    if not p.exists():
        pytest.skip("Args pickle not found: {}".format(p))
    return str(p)


@pytest.fixture(scope="session")
def data_pickle(data_dir):
    """Path to ``vesicle_data.pickle``, or *None* if unavailable.

    When the user supplies a custom ``--data-dir`` that was produced on a
    different machine, ``vesicle_data.pickle`` (which stores the expected
    assertion values) may not exist.  In that case this fixture returns
    *None* and value-comparison tests are automatically skipped.
    """
    p = data_dir / "vesicle_data.pickle"
    if not p.exists():
        return None
    return str(p)
