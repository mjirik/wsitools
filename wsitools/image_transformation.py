from loguru import logger
import skimage.filters
from skimage.morphology import skeletonize
import skimage.io
import scipy.signal
import scipy.ndimage
import os.path as op
import numpy as np
import warnings
import morphsnakes as ms
from matplotlib import pyplot as plt
from scaffan import image as scim
import scaffan
import scaffan.lobulus
import scaffan.image
from exsu.report import Report
from pyqtgraph.parametertree import Parameter
import imma.image
import copy
from typing import Optional


class ImageExport:
    def __init__(
        self,
        pname="Image Export",
        ptype="bool",
        pvalue=False,
        ptip="Transform input image to other image format",
        report: Optional[Report] = None,
    ):
        params = [
            # {
            #     "name": "Tile Size",
            #     "type": "int",
            #     "value" : 128
            # },
            {
                "name": "Working Resolution",
                "type": "float",
                "value": 0.000001,  # 10 µm
                # "value": 0.00000091,  # this is typical resolution on level 2
                # "value": 0.00000091,  # this is typical resolution on level 2
                # "value": 0.00000182,  # this is typical resolution on level 3
                # "value": 0.00000364,  # this is typical resolution on level 4
                # "value": 0.00000728,  # this is typical resolution on level 5
                # "value": 0.00001456,  # this is typical resolution on level 6
                "suffix": "m",
                "siPrefix": True,
            },
            {
                "name": "File Format",
                "type": "str",
                "value": "tiff",
                # "suffix": "m",
                "siPrefix": False,
                "tip": "JPG, PNG and tiff format are allowed",
            },
        ]

        self.parameters = Parameter.create(
            name=pname,
            type=ptype,
            value=pvalue,
            tip=ptip,
            children=params,
            expanded=False,
        )
        self.anim: Optional[scaffan.image.AnnotatedImage] = None
        self.report: Report = report

    # def set_report(self, report: Report):
    #     self.report = report

    def run(self, anim: scaffan.image.AnnotatedImage):
        self.anim = anim
        pxsz_mm = self.parameters.param("Working Resolution").value() * 1000
        ff = str(self.parameters.param("File Format").value())

        view = self.anim.get_full_view(pixelsize_mm=pxsz_mm)
        imrgb = view.get_raster_image(as_gray=False)
        # self.report.outputdir
        self.report.imsave(f"whole_scan_export.{ff}", imrgb)
