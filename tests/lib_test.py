import pytest
from loguru import logger
# from wsitools import libfixer


def test_import_openslide():
    import openslide_bin
    import openslide
    # check openslide version
    assert (openslide_bin.__version__).startswith("4.")
    assert (openslide.__version__).startswith("1.")

