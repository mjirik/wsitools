# /usr/bin/env python
# -*- coding: utf-8 -*-
from loguru import logger
import click.testing
import shutil
import pytest
import scaffan.main_cli
import io3d
from pathlib import Path

from scaffan import image_intensity_rescale
import skimage.data
from matplotlib import pyplot as plt
import numpy as np


def test_image_intensity_rescale():
    # img = skimage.data.astronaut()
    # img = skimage.data.chelsea()
    img = np.random.rand(*[120, 80, 3])
    img[10:30, 30:50, :] += 0.4
    img[50:70, 10:40, 0] += 0.6
    img = (img * 255).astype(np.uint8)

    rescaler = image_intensity_rescale.RescaleIntensityPercentile()
    rescaler.calculate_intensity_dependent_parameters(img)
    img_rescaled = rescaler.rescale_intensity(img)
    # plt.figure()
    # plt.imshow(img)
    # plt.figure()
    # plt.imshow(img_rescaled)

    assert img.shape[0] == img_rescaled.shape[0]
    assert img.shape[1] == img_rescaled.shape[1]
    assert img.shape[2] == img_rescaled.shape[2]
    hsty, hstx = np.histogram(img.ravel(), bins=20)
    hstyr, hstxr = np.histogram(img_rescaled.ravel(), bins=20)
    # plt.figure()
    # plt.plot(hstx[1:], hsty)
    # plt.figure()
    # plt.plot(hstxr[1:], hstyr)
    # plt.show()
    # the rescaled histogram should be flatter
    assert np.max(hstx) > np.max(hstxr)

    # imthumb = imsl.get_thumbnail((512, 512))
    # pth = io3d.datasets.join_path(
    #     "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    # )
    #
    # logger.debug(f"pth={pth}, exists={Path(pth).exists()}")
    # expected_pth = Path(".test_output/data.xlsx")
    # logger.debug(f"expected_pth={expected_pth}, exists: {expected_pth.exists()}")
    # if expected_pth.exists():
    #     shutil.rmtree(expected_pth.parent)
    #
    # runner = click.testing.CliRunner()
    # # runner.invoke(anwa.main_click.nogui, ["-i", str(pth)])
    # runner.invoke(
    #     scaffan.main_cli.run,
    #     ["nogui", "-i", pth, "-o", expected_pth.parent, "-c", "#FFFF00"],
    # )
    #
    # assert expected_pth.exists()
