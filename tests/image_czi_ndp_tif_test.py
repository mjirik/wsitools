from loguru import logger
import unittest
import os.path as op
from pathlib import Path
import pytest
import skimage.transform

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

# import wsitools.libfixer

# wsitools.libfixer.import_openslide()
import openslide_bin
import openslide
import io3d
import wsitools.image as scim


def _check_landscape(fn):
    """
    Read landscape image and check the size and shape.

    The part of the test is also reading smaller part of the
    image from the middle top of the full image and prepare it for visual check.
    """

    fn = io3d.joinp(r"medical\orig\Scaffan-analysis-czi\J7_5\J7_5_b_test_4_landscape.czi")
    # fn = io3d.joinp("medical/orig/scaffan_png_tiff/split_176_landscape.tif")
    # fn = io3d.joinp(r"medical/orig/sample_data/SCP003/SCP003.ndpi")
    # fn.absolute()
    assert Path(fn).absolute().exists()
    logger.debug("filename {}".format(fn))
    pixelsize_mm = [0.01, 0.01]
    anim = scim.AnnotatedImage(fn)
    view = anim.get_full_view(pixelsize_mm=pixelsize_mm)
    img = view.get_raster_image()
    size = view.get_size_on_pixelsize_mm()
    print(img.shape)
    plt.imshow(img)
    # plt.show()
    properties = anim.openslide.properties

    assert size[0] > size[1]
    assert img.shape[0] < img.shape[1]

    # get smaller view from the middle of the top of the image
    sz_px = img.shape[0] / 3
    size_mm = [sz_px * pixelsize_mm[0]*0.5, sz_px * pixelsize_mm[1]]
    location_mm = [img.shape[1] * pixelsize_mm[1] * 1. / 3., 0]
    view_small = anim.get_view(
        location_mm=location_mm,
            size_mm=size_mm,
            pixelsize_mm=pixelsize_mm
        )
    img_small = view_small.get_raster_image()
    # plt.imshow(img_small)
    # plt.suptitle("This should middle top part of the full image")
    # plt.show()

    assert img_small.shape[0] > img_small.shape[1]

    # get top middle part of the img

    # now the 0-th is vertical and 1-th is horizontal
    img_small_0 = img[
        0:0+img_small.shape[0],
        int(img.shape[1] * 1. / 3.):int(img.shape[1] * 1. / 3.) + img_small.shape[1],
        :
    ]

    assert img_small.shape == img_small_0.shape

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img_small)
    plt.title("get_view()")
    plt.subplot(1,2,2)
    plt.imshow(img_small_0)
    plt.title("slice from the full image")
    plt.suptitle("Images should be the same")
    plt.show()

    # compare quadrants of the images
    sz = [8, 8, 3]
    diff = np.sum(np.abs(
        skimage.transform.resize(img_small, sz) - skimage.transform.resize(img_small_0, sz)
    ))
    assert diff < 5., "The sub-image seems to be not the same."

def test_landscape_czi():
    fn = io3d.joinp(r"medical\orig\Scaffan-analysis-czi\J7_5\J7_5_b_test_4_landscape.czi")
    _check_landscape(fn)

def test_landscape_tiff():
    fn = io3d.joinp("medical/orig/scaffan_png_tiff/split_176_landscape.tif")
    _check_landscape(fn)

def test_landscape_ndpi():
    fn = io3d.joinp(r"medical/orig/sample_data/SCP003/SCP003.ndpi")
    _check_landscape(fn)
