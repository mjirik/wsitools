from loguru import logger
import unittest
import os.path as op
from pathlib import Path
import pytest

# path_to_script = op.dirname(op.abspath(__file__))
path_to_script = Path(__file__).absolute().parent

import sys

sys.path.insert(0, op.abspath(op.join(path_to_script, "../../io3d")))
sys.path.insert(0, op.abspath(op.join(path_to_script, "../../imma")))
# import sys
# import os.path

# imcut_path =  os.path.join(path_to_script, "../../imcut/")
# sys.path.insert(0, imcut_path)

import glob
import os
import numpy as np
import matplotlib.pyplot as plt

skip_on_local = False

import wsitools.libfixer

# wsitools.libfixer.import_openslide()
import openslide_bin
import openslide
import io3d
import wsitools.image as scim

def test_read_tiff_landscape():

    # fn = io3d.joinp(r"medical\orig\Scaffan-analysis-czi\J7_5\J7_5_b_test_4_landscape.czi")
    # fn = io3d.joinp("medical/orig/scaffan_png_tiff/split_176_landscape.tif")
    fn = io3d.joinp(r"medical/orig/sample_data/SCP003/SCP003.ndpi")
    # fn.absolute()
    assert Path(fn).absolute().exists()
    logger.debug("filename {}".format(fn))
    anim = scim.AnnotatedImage(fn)
    view = anim.get_full_view(pixelsize_mm=[0.01, 0.01])
    img = view.get_raster_image()
    size = view.get_size_on_pixelsize_mm()
    print(img.shape)
    plt.imshow(img)
    plt.show()
    properties = anim.openslide.properties
    assert size[0] < size[1]
    assert img.shape[0] < img.shape[1]
