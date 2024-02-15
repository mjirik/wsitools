#! /usr/bin/python
# -*- coding: utf-8 -*-

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

import scaffan.image as scim

scim.import_openslide()
import openslide
import io3d


def test_read_tiff():

    fn = io3d.datasets.join_path("medical/orig/biodur_sample/0001.tiff", get_root=True)
    logger.debug("filename {}".format(fn))
    anim = scim.AnnotatedImage(fn)
    size_px = 100
    size_on_level = [size_px, size_px]
    view = anim.get_view(
        location_mm=[10, 11],
        size_on_level=size_on_level,
        level=5,
        # size_mm=[0.1, 0.1]
    )
    img = view.get_region_image(as_gray=True)
    # plt.imshow(img)
    # plt.show()
    assert img.shape[0] == size_px
    assert img.shape[1] == size_px
    # offset = anim.get_offset_px()
    # assert len(offset) == 2, "should be 2D"
    im = anim.get_image_by_center((100, 100), as_gray=True)
    assert len(im.shape) == 2, "should be 2D"

    annotations = anim.read_annotations()
    assert len(annotations) == 0, "there should be 2 annotations"
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # assert im[0, 0] == pytest.approx(
    #     0.767964705882353, 0.001
    # )  # expected intensity is 0.76
    # imsl = openslide.OpenSlide(fn)
    imsl = anim.openslide

    pixelsize1, pixelunit1 = scim.get_pixelsize(imsl)
    assert pixelsize1[0] > 0
    assert pixelsize1[1] > 0

    pixelsize2, pixelunit2 = scim.get_pixelsize(imsl, level=2)
    assert pixelsize2[0] > 0
    assert pixelsize2[1] > 0

    assert pixelsize2[0] > pixelsize1[0]
    assert pixelsize2[1] > pixelsize1[1]
