"""Smoke tests to verify that core dependencies are installed and importable."""

import importlib


def _check_import(module_name: str) -> str:
    """Import a module by name and return its version string.

    Args:
        module_name: Fully qualified Python module name.

    Returns:
        The version string reported by the module.

    Raises:
        ImportError: If the module cannot be imported.
        AttributeError: If the module has no ``__version__`` attribute.
    """
    mod = importlib.import_module(module_name)
    return mod.__version__


def test_numpy_importable() -> None:
    """Numpy must be importable and expose a version."""
    version = _check_import("numpy")
    assert isinstance(version, str)


def test_opencv_importable() -> None:
    """OpenCV must be importable and expose a version."""
    version = _check_import("cv2")
    assert isinstance(version, str)


def test_ultralytics_importable() -> None:
    """Ultralytics must be importable and expose a version."""
    version = _check_import("ultralytics")
    assert isinstance(version, str)


def test_sklearn_importable() -> None:
    """Scikit-learn must be importable and expose a version."""
    version = _check_import("sklearn")
    assert isinstance(version, str)
