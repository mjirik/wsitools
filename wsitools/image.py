# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for slice image and view processing. It cooperates with openslide.

Be careful because some of coordinates are swapped. It is due to openslide.
With request subimage with size=[A, B] it will return subimage with shape [B, A]. It is because of visualization.

There are two main structures. AnnotatedImage and View.


"""

from loguru import logger

# problem is loading lxml together with openslide
# from lxml import etree
from typing import List, Union
import os.path as op
import glob
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import skimage.color
import skimage.io
import skimage.transform
from . import annotation as scan
from . import libfixer
from . import image_intensity_rescale
import imma
import imma.image
from pathlib import Path

from matplotlib.path import Path as mplPath
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from PIL.TiffTags import TAGS
import traceback
import re

annotationID = Union[int, str]
annotationIDs = List[annotationID]


import openslide_bin
import openslide

# libfixer.import_openslide()
# try:
#     import openslide
# except ImportError as e:
#     logger.debug("Cannot import openslide. Use conda install -c conda-forge openslide-python")
#     raise e



def fix_location_ndpi(imsl, location, level):
    return list(
        (location - np.mod(location, imsl.level_downsamples[level])).astype(int)
    )


def fix_location_czi(imsl, location, level):
    return list(np.asarray(location) - imsl._czi_start)
    # return list(
    #     (location - np.mod(location, imsl.level_downsamples[level])).astype(int)
    # )


# def
def get_image_by_center(
    imsl, center, level=3, size=None, as_gray=True, do_fix_location=True
):
    if size is None:
        size = np.array([800, 800])

    location = get_region_location_by_center(imsl, center, level, size)
    if do_fix_location:
        location = fix_location_ndpi(imsl, location, level)
    imcr = imsl.read_region(location, level=level, size=size)
    im = np.asarray(imcr)
    if as_gray:
        im = skimage.color.rgb2gray(im)
    return im


def get_region_location_by_center(imsl, center, level, size):
    size2 = (np.asarray(size) / 2).astype(int)

    offset = size2 * imsl.level_downsamples[level]
    location = (np.asarray(center) - offset).astype(int)
    return location


def get_region_center_by_location(imsl, location, level, size):
    size2 = (np.asarray(size) / 2).astype(int)

    offset = size2 * imsl.level_downsamples[level]
    center = (np.asarray(location) + offset).astype(int)
    return center


def get_pixelsize(imsl, level=0, requested_unit="mm"):
    """
    imageslice
    :param imsl: image slice obtained by openslice.OpenSlide(path)
    :return: pixelsize, pixelunit
    """
    pm = imsl.properties
    resolution_unit = pm.get("tiff.ResolutionUnit")
    resolution_x = pm.get("tiff.XResolution")
    resolution_y = pm.get("tiff.YResolution")
    logger.trace(f"Resolution {resolution_x}x{resolution_y} pixels/{resolution_unit}")
    downsamples = imsl.level_downsamples[level]

    pixelunit = None

    input_resolution_unit = resolution_unit
    if resolution_unit is None:
        pixelunit = resolution_unit
    elif requested_unit in ("mm"):
        if resolution_unit in ("mm", "milimeters"):
            pixelunit = "mm"
        elif resolution_unit in ("cm", "centimeter"):
            downsamples = downsamples * 10.0
            pixelunit = "mm"
        elif resolution_unit in ("m"):
            downsamples = downsamples * 1000.0
            pixelunit = "mm"

        else:
            raise ValueError(
                "Cannot covert from {} to {}.".format(
                    input_resolution_unit, requested_unit
                )
            )
    else:
        raise ValueError(
            "Cannot covert from {} to {}.".format(input_resolution_unit, requested_unit)
        )

    # if resolution_unit != resolution_unit:
    #     raise ValueError("Cannot covert from {} to {}.".format(input_resolution_unit, requested_unit))

    pixelsize = np.asarray(
        [downsamples / float(resolution_x), downsamples / float(resolution_y)]
    )

    logger.trace(f"pixelsize={pixelsize}, pixelunit={pixelunit}")
    return pixelsize, pixelunit


# def annoatation_px_to_mm(imsl: openslide.OpenSlide, annotation: dict) -> dict:


def get_resize_parameters(imsl, former_level, former_size, new_level):
    """
    Get scale factor and size of image on different level.

    :param imsl: OpenSlide
    :param former_level: int
    :param former_size: list of ints
    :param new_level: int
    :return: scale_factor, new_size
    """
    scale_factor = (
        imsl.level_downsamples[former_level] / imsl.level_downsamples[new_level]
    )
    new_size = (np.asarray(former_size) * scale_factor).astype(int)
    return scale_factor, new_size


class ImageSlide:
    """
    Same interface as OpenSlide for images with other format (.tiff and .czi).
    """

    def __init__(self, path):
        self.openslide: openslide.OpenSlide = None
        self.path = path
        self.imagedata = None
        self.properties = None
        self.compatible_with_openslide = True
        if Path(self.path).suffix.lower() in (".tiff", ".tif"):
            self.image_type = ".tiff"
            self._get_imagedata = self._get_imagedata_tiff
            self.get_thumbnail = self._get_thumbnail_tiff
            self.read_region = self._read_region_other_than_ndpi
            self._set_properties_tiff()
            self.level_downsamples = [float(2 ** i) for i in range(0, 8)]
            self.level_count = len(self.level_downsamples)

        if Path(self.path).suffix.lower() in (".czi"):
            self.image_type = ".czi"
            self.get_thumbnail = self._get_thumbnail_czi
            self._get_imagedata = self._get_imagedata_czi
            # self.read_region = self._read_region_other_than_ndpi
            self.read_region = self._read_region_nzi
            self._set_properties_czi()
            self.level_downsamples = [float(2 ** i) for i in range(0, 8)]
            self.level_count = len(self.level_downsamples)
            # self.dimensions = asdfimsl.dimensions

        # if Path(self.path).suffix.lower() in (".czi"):
        #     self.image_type = ".czi"
        #     self.get_thumbnail = self._get_thumbnail_czi
        #     self._get_imagedata = self._get_imagedata_czi
        #     self.read_region = self._read_region_other_than_ndpi
        #     self._set_properties_czi()
        #     self.level_downsamples = [float(2 ** i) for i in range(0, 8)]
        #     self.level_count = len(self.level_downsamples)

        elif Path(self.path).suffix.lower() in (".ndpi"):
            self.image_type = ".ndpi"
            imsl = openslide.OpenSlide(path)
            self.openslide = imsl
            self.properties = imsl.properties
            self.level_downsamples = imsl.level_downsamples
            self.dimensions = imsl.dimensions
            self.level_count = imsl.level_count
            self.get_thumbnail = imsl.get_thumbnail
            self.read_region = imsl.read_region

    def _get_thumbnail_tiff(self, size):
        self._get_imagedata_tiff()
        return skimage.transform.resize(self.imagedata, size)

    def _get_imagedata_tiff(self):
        if self.imagedata is None:
            self.imagedata = skimage.io.imread(self.path)
        return self.imagedata

    def _get_thumbnail_czi(self, size):
        self._get_imagedata_czi()
        return skimage.transform.resize(self.imagedata, size)

    def _get_imagedata_czi(self):
        from czifile import CziFile

        if self.imagedata is None:
            with CziFile(self.path) as czi:
                image_arrays = czi.asarray()
                metadata = czi.metadata()
            img = np.squeeze(image_arrays)
            if img.ndim != 3:
                msg = f"Wrong input data dimension. Expected 3, got {img.ndim}."
                logger.error(msg)
                raise Exception(msg)
            self.imagedata = img
        return self.imagedata

    def _get_metadata_czi(self, picture_path_annotation):
        from czifile import CziFile

        with CziFile(picture_path_annotation) as czi:
            metadata = czi.metadata()
        return metadata

    def _read_region_other_than_ndpi(self, location, level, size):
        """
        Works also for small nzi files

        :param location:
        :param level:
        :param size:
        :return:
        """
        img = self._get_imagedata()
        factor = int(2 ** level)
        # factor = self.level_downsamples[level]

        newshape = list(img.shape)
        newshape[0] = size[1]
        newshape[1] = size[0]
        sl1 = slice(location[0], location[0] + (size[0] * factor))
        sl0 = slice(location[1], location[1] + (size[1] * factor))
        imcrop = img[sl0, sl1].copy()
        logger.debug(f"imcrop.shape={imcrop.shape}, newshape={newshape}")
        out = skimage.transform.resize(imcrop, newshape)
        return out

    def _read_region_nzi(self, location, level, size):
        from czifile import CziFile
        from . import image_czi

        image_czi.instal_codecs_with_pip()

        # This swap make all the behavior compatible with OpenSlide
        # TODO maybe here is the problem
        if self.compatible_with_openslide:
            # swap axes
            location = [location[1], location[0]]
            size = [size[1], size[0]]

        factor = int(2 ** level)
        # factor = self.level_downsamples[level]

        with CziFile(self.path) as czi:

            location_fixed = np.asarray(location) + self._czi_start
            # self._czi_start = czi.start[-3:-1]
            output = image_czi.read_region_with_level(
                czi, location_fixed, size, level=level
            )
        return output

    def _set_properties_czi(self):
        import xml
        import xml.etree.ElementTree as ET
        from czifile import CziFile

        # from lxml import etree

        with CziFile(self.path) as czi:
            metadata = czi.metadata()
            self._czi_start = czi.start[-3:-1]
            self._czi_shape = czi.shape[-3:-1]
            self.dimensions = self._czi_shape
        # root = etree.fromstring(metadata)
        # xres = float(root.xpath('//Distance[@Id="X"]/Value')[0].text)
        # yres = float(root.xpath('//Distance[@Id="Y"]/Value')[0].text)
        root = ET.fromstring(metadata)
        xres = float(root.findall('.//Distance[@Id="X"]/Value')[0].text)
        yres = float(root.findall('.//Distance[@Id="Y"]/Value')[0].text)
        # xx = float(root.findall('.//Translate[@Id="X"]/Value')[0].text)
        # yy = float(root.findall('.//Translate[@Id="Y"]/Value')[0].text)

        # take care about cropped images created by Zen Blue
        # TODO use translate
        # translate = root.findall('.//Translate')
        # if len(translate) > 0:
        #     coord0 = translate[0].attrib['X']
        #     coord1 = translate[0].attrib['Y']
        #     start = list(self._czi_start)
        # start[0]
        # start[0] += -float(coord0)
        # start[1] += -float(coord1)
        # self._czi_start = tuple(start)
        # yy = root.findall('.//Translate')[0].attrib['Y']
        meta_dict = {}
        logger.debug(f"nzi pixelsize  {xres}x{yres} [m]")
        meta_dict["tiff.ResolutionUnit"] = "m"
        meta_dict["tiff.XResolution"] = 1 / xres
        meta_dict["tiff.YResolution"] = 1 / yres
        # TODO here it might be swaped
        meta_dict["openslide.level[0].height"] = self._czi_shape[0]
        meta_dict["openslide.level[0].width"] = self._czi_shape[1]
        meta_dict["hamamatsu.XOffsetFromSlideCentre"] = -int(self._czi_shape[0] / 2)
        meta_dict["hamamatsu.YOffsetFromSlideCentre"] = -int(self._czi_shape[1] / 2)
        # meta_dict["hamamatsu.XOffsetFromSlideCentre"] = 0
        # meta_dict["hamamatsu.YOffsetFromSlideCentre"] = 0
        self.properties = meta_dict

    def _set_properties_tiff(self):
        with Image.open(self.path) as img:
            meta_dict = {TAGS[key]: img.tag[key] for key in img.tag}

        unit_multiplicator = 1
        if "ImageDescription" in meta_dict:
            key_value = [
                couplestring.split("=")
                for couplestring in meta_dict["ImageDescription"][0].split("\n")
            ]
            image_description = {kv[0]: kv[1] for kv in key_value if len(kv) > 1}

            if "unit" in image_description:
                if image_description["unit"] == "micron":
                    unit_multiplicator = 0.000001

        try:
            xr = meta_dict["XResolution"]
            logger.debug(f"xr={xr}")
            xres = (xr[0][1] / xr[0][0]) * unit_multiplicator
            # self.parameters.param("Input", "Pixel Size X").setValue(xres)
            yr = meta_dict["YResolution"]
            logger.debug(f"yr={xr}")
            yres = (yr[0][1] / yr[0][0]) * unit_multiplicator
        except Exception as e:
            logger.debug(traceback.format_exc())
            logger.warning("Resolution reading failed")
            xres = 1.0
            yres = 1.0

        # TODO
        self.dimensions = [meta_dict["ImageLength"][0], meta_dict["ImageWidth"][0]]
        meta_dict["tiff.ResolutionUnit"] = "m"
        meta_dict["tiff.XResolution"] = 1.0 / xres
        meta_dict["tiff.YResolution"] = 1.0 / yres
        meta_dict["hamamatsu.XOffsetFromSlideCentre"] = 0
        meta_dict["hamamatsu.YOffsetFromSlideCentre"] = 0
        # TODO width is set to length here
        meta_dict["openslide.level[0].height"] = self.dimensions[0]
        meta_dict["openslide.level[0].width"] = self.dimensions[1]
        self.properties = meta_dict

    #     if self.openslide is not None:
    #         return self.openslide.get_thumbnail(size)
    #
    # def read_region(self, location, level, size):
    #     if self.openslide is not None:
    #         return self.openslide.get_thumbnail(size)


class AnnotatedImage:
    """
    Read the image and the annotation. The
    """

    def __init__(self, path: str, skip_read_annotations=False):
        fs_enc = sys.getfilesystemencoding()
        logger.debug(f"fs_enc: {fs_enc}")
        logger.debug("Reading file {}".format(path))

        self.path = path
        # pth_encoded = path.encode(fs_enc)
        # path.encode()
        # logger.debug(f"path encoded {pth_encoded}")
        self.image_type: str = Path(self.path).suffix.lower()
        if self.image_type in (".tiff", ".tif"):
            self.image_type = ".tiff"
        self.openslide: ImageSlide = ImageSlide(path)
        self.region_location = None
        self.region_size = None
        self.region_level = None
        self.region_pixelsize = None
        self.region_pixelunit = None
        self.pixelunit = "mm"
        self.level_pixelsize: list = None
        self.level_pixelsize_derived_from_resolution_on_level_0: bool = False
        self._set_level_pixelsize()
        self.annotations = None
        if not skip_read_annotations:
            self.read_annotations()
        self.intensity_rescaler = image_intensity_rescale.RescaleIntensityPercentile()
        self.run_intensity_rescale = False
        self.raster_image_preprocessing_function_handler = []  # you can add

    def _run_raster_image_preprocessing_function_handler(self, img):

        for fcn in self.raster_image_preprocessing_function_handler:
            img = fcn(img)
        return img

    def update_pixelsize(self, pixelsize_on_level_0):
        """
        Used to change pixelsize from GUI.
        :param pixelsize_on_level_0:
        :return:
        """
        self._set_level_pixelsize(pixelsize_on_level_0)
        self.read_annotations()

    def _set_level_pixelsize(self, pixelsize_on_level_0=None):
        """

        :param pixelsize_on_level_0: list or array of two numbers
        :return:
        """
        if pixelsize_on_level_0 is not None:
            pixelsize_on_level_0 = np.asarray(pixelsize_on_level_0)
            self.level_pixelsize = [
                pixelsize_on_level_0 / float(2 ** i) for i in range(0, 7)
            ]
            self.level_pixelsize_derived_from_resolution_on_level_0 = True
        else:
            self._set_level_pixelsize_with_openslide()

    def _set_level_pixelsize_with_openslide(self):

        self.level_pixelsize = [
            get_pixelsize(self.openslide, i, requested_unit=self.pixelunit)[0]
            for i in range(0, self.openslide.level_count)
        ]

    def set_intensity_rescale_parameters(
        self,
        run_intensity_rescale=False,
        percentile_range=(5, 95),
        percentile_map_range=(-0.9, 0.9),
        sig_slope=1,
    ):
        self.intensity_rescaler.set_parameters(
            percentile_range=percentile_range,
            percentile_map_range=percentile_map_range,
            sig_slope=sig_slope,
        )

        self.run_intensity_rescale = run_intensity_rescale
        if self.run_intensity_rescale:
            self.update_rescale_parameters()

    def update_rescale_parameters(self):
        img = np.array(self.openslide.get_thumbnail((512, 512)))
        self.intensity_rescaler.calculate_intensity_dependent_parameters(img)

    def get_file_info(self):
        pxsz, unit = self.get_pixel_size(0)
        # self.titles
        # self.colors
        return "Pixelsize: {}x{} [{}], {} annotations".format(
            pxsz[0], pxsz[1], unit, len(self.id_by_colors)
        )

    def get_optimal_level_for_fluent_resize(self, pixelsize_mm, safety_bound=2):
        if np.isscalar(pixelsize_mm):
            pixelsize_mm = [pixelsize_mm, pixelsize_mm]
        pixelsize_mm = np.asarray(pixelsize_mm)

        pixelsize_mm2 = pixelsize_mm / safety_bound
        best_level = 0
        # scale_factor = None
        for i, pxsz in enumerate(self.level_pixelsize):
            if np.array_equal(pxsz, pixelsize_mm):
                best_level = i
                # scale_factor = 1.0
            elif all(pxsz < pixelsize_mm2):
                best_level = i

        return best_level

    def get_resize_parameters(self, former_level, former_size, new_level):
        """
        Get scale and size of image after resize to other level

        :param former_level:
        :param former_size:
        :param new_level:
        :return: scale_factor, new_size
        """
        return get_resize_parameters(
            self.openslide, former_level, former_size, new_level
        )

    def get_offset_px(self):
        return get_offset_px(self.openslide)

    def get_pixel_size(self, level=0):
        return self.level_pixelsize[level], self.pixelunit
        # return get_pixelsize(self.openslide, level)

    def get_slide_size(self):
        height0 = self.openslide.properties["openslide.level[0].height"]
        width0 = self.openslide.properties["openslide.level[0].width"]
        size_on_level0 = np.array([int(height0), int(width0)])
        return size_on_level0

    def get_image_by_center(self, center, level=3, size=None, as_gray=True):
        do_fix_location = True if self.image_type == ".ndpi" else False
        img = get_image_by_center(
            self.openslide,
            center,
            level,
            size,
            as_gray,
            do_fix_location=do_fix_location,
        )
        img = self._run_raster_image_preprocessing_function_handler(img)
        if self.run_intensity_rescale:
            img = self.intensity_rescaler.rescale_intensity(img)
        return img

    def get_region_location_by_center(self, center, level, size):
        return get_region_location_by_center(self.openslide, center, level, size)

    def get_region_center_by_location(self, location, level, size):
        return get_region_center_by_location(self.openslide, location, level, size)

    def load_zeiss_elements(self, metadata):
        from xml.dom import minidom

        pixelSizeMM = self.get_pixel_size()
        root = minidom.parseString(metadata)
        elements = root.getElementsByTagName("Elements")

        listOfBeziers = []
        listOfBeziersNames = []
        listOfBeziersColors = []
        listOfCircles = []
        listOfRectangles = []
        listOfEllipses = []

        for j in range(len(elements)):
            for child in elements[j].childNodes:
                if child.nodeName == "Bezier":
                    listOfPoints_temp = []
                    points = child.getElementsByTagName("Points")[
                        0
                    ].firstChild.nodeValue
                    temp_points = points.split()
                    for i in range(len(temp_points)):
                        point_X = (
                            float(temp_points[i].split(",")[0]) * pixelSizeMM[0][0]
                        )
                        point_Y = (
                            float(temp_points[i].split(",")[1]) * pixelSizeMM[0][1]
                        )

                        pointsXY_float = (point_X, point_Y)
                        listOfPoints_temp.append(pointsXY_float)
                        if i == 0:
                            lastPointsXY = pointsXY_float  # aby byla krivka spojena
                    listOfPoints_temp.append(lastPointsXY)
                    listOfBeziers.append(listOfPoints_temp)
                    name = child.getElementsByTagName("Name")[0].firstChild.nodeValue
                    listOfBeziersNames.append(name)
                    stroke = child.getElementsByTagName("Stroke")
                    if len(stroke) > 0:

                        # remove alpha:   #FFFF0000 -> #FF0000
                        color = "#" + stroke[0].firstChild.nodeValue[-6:]
                    else:
                        color = "#FF0000"
                    listOfBeziersColors.append(color)

                elif child.nodeName == "Circle":
                    center_X = (
                        float(child.getElementsByTagName("CenterX")[0].firstChild.data)
                        * pixelSizeMM[0][0]
                    )  # v mm od leveho okraje
                    center_Y = (
                        float(child.getElementsByTagName("CenterY")[0].firstChild.data)
                        * pixelSizeMM[0][1]
                    )  # v mm od horniho okraje obrazku
                    radius = (
                        float(child.getElementsByTagName("Radius")[0].firstChild.data)
                        * pixelSizeMM[0][0]
                    )
                    listOfCircles.append((center_X, center_Y, radius))

                elif child.nodeName == "Rectangle":
                    X_left_top = (
                        float(child.getElementsByTagName("Left")[0].firstChild.data)
                        * pixelSizeMM[0][0]
                    )
                    Y_left_top = (
                        float(child.getElementsByTagName("Top")[0].firstChild.data)
                        * pixelSizeMM[0][1]
                    )
                    width_rec = (
                        float(child.getElementsByTagName("Width")[0].firstChild.data)
                        * pixelSizeMM[0][0]
                    )
                    height_rec = (
                        float(child.getElementsByTagName("Height")[0].firstChild.data)
                        * pixelSizeMM[0][1]
                    )
                    listOfRectangles.append(
                        (X_left_top, Y_left_top, width_rec, height_rec)
                    )

                elif child.nodeName == "Ellipse":
                    center_X = (
                        float(child.getElementsByTagName("CenterX")[0].firstChild.data)
                        * pixelSizeMM[0][0]
                    )
                    center_Y = (
                        float(child.getElementsByTagName("CenterY")[0].firstChild.data)
                        * pixelSizeMM[0][1]
                    )
                    radiusX = (
                        float(child.getElementsByTagName("RadiusX")[0].firstChild.data)
                        * pixelSizeMM[0][0]
                    )
                    radiusY = (
                        float(child.getElementsByTagName("RadiusY")[0].firstChild.data)
                        * pixelSizeMM[0][1]
                    )
                    listOfEllipses.append((center_X, center_Y, radiusX, radiusY))

        return (
            listOfBeziers,
            listOfCircles,
            listOfRectangles,
            listOfEllipses,
            listOfBeziersNames,
            listOfBeziersColors,
        )

    def insert_zeiss_annotation_bezier(
        self, anim, listOfBeziers, listOfBeziersNames, listOfBeziersColors
    ):
        pixelSizeMM = anim.get_pixel_size()
        if len(listOfBeziers) != 0:
            anim.annotations = []
            for bezier, name, color in zip(
                listOfBeziers, listOfBeziersNames, listOfBeziersColors
            ):
                x_mm = []
                y_mm = []
                for tuple_XY in bezier:
                    x_mm.append(tuple_XY[0])
                    y_mm.append(tuple_XY[1])

                x_px = np.asarray(x_mm) / pixelSizeMM[0][0]
                y_px = np.asarray(y_mm) / pixelSizeMM[0][1]

                anim.annotations.append(
                    {
                        "x_mm": x_mm,
                        "y_mm": y_mm,
                        "color": color,
                        "x_px": x_px,
                        "y_px": y_px,
                        "title": name,
                        "details": "",
                    }
                )

                # views = anim.get_views([0], pixelsize_mm = [0.01, 0.01])
            # views = anim.get_views(*args, **kwargs) # vybiram, jakou chci zobrazit anotaci
            # view = views[0]
            # img = view.get_region_image(as_gray = False)
            # plt.imshow(img)
            # view.plot_annotations()
            # plt.show()

    def read_annotations(self):
        """
        Read all annotations of the file and save extracted information.
        :return:
        """
        logger.debug(f"Reading the annotation {self.path}")
        self.annotations = []
        if self.image_type == ".ndpi":
            self.annotations = scan.read_annotations_ndpa(self.path)
            self.annotations = scan.annotations_to_px(self.openslide, self.annotations)
        elif self.image_type == ".czi":
            self.annotations = []
            # TODO nastavení self.annotations na základě anim
            #  self.path # cesta k CZI souboru

            metadata_czi = self.openslide._get_metadata_czi(
                self.path
            )  # nacitani metadat pomoci vlastni metody (asi bude fungovat spise)
            (
                listOfBeziers,
                listOfCircles,
                listOfRectangles,
                listOfEllipses,
                listOfBeziersNames,
                listOfBeziersColors,
            ) = self.load_zeiss_elements(metadata=metadata_czi)
            #  self.annotations = insert_zeiss_annotation_bezier(anim=self, ...)
            self.insert_zeiss_annotation_bezier(
                anim=self,
                listOfBeziers=listOfBeziers,
                listOfBeziersNames=listOfBeziersNames,
                listOfBeziersColors=listOfBeziersColors,
            )
            #  test function tests / image_czi_test.py

        # here can be also reading of imagej annotations in all the cases something like if len(self.annotations) == 0:
        else:  # if self.image_type == ".tiff":
            slide_size = self.get_slide_size()
            # pxsz, unit = self.get_pixel_size(0)
            self.annotations = scan.read_annotations_imagej(
                self.path, slide_size=slide_size
            )
            self.annotations = scan.annotations_px_to_mm(
                self.openslide, self.annotations
            )

            # self.annotations = scan.annotations(self.openslide, self.annotations)
        self.id_by_titles = scan.annotation_titles(self.annotations)
        self.id_by_colors = scan.annotation_colors(self.annotations)
        # self.details = scan.annotation_details(self.annotations)
        return self.annotations

    def get_annotation_center_mm(self, ann_id: Union[str, int]):
        ann_id = self.get_annotation_id(ann_id)
        ann = self.annotations[ann_id]
        return (np.mean(ann["x_mm"]), np.mean(ann["y_mm"]))

    def get_full_view(
        self,
        level=0,
        pixelsize_mm=None,
        safety_bound=2,
        margin=0.0,
        # margin_in_pixels=False,
        # annotation_id=None,
        # margin=0.5,
        # margin_in_pixels=False,
    ) -> "View":

        height0 = self.openslide.properties["openslide.level[0].height"]
        width0 = self.openslide.properties["openslide.level[0].width"]
        # TODO check if this is correct
        # size_on_level = np.array([int(height0), int(width0)])
        size_on_level = np.array([int(width0), int(height0)])
        # size_on_level = [width, height]
        view_level0 = self.get_view(
            location=[0, 0],
            level=0,
            size_on_level=size_on_level,
            margin=margin,
            margin_in_pixels=False,
        )
        if pixelsize_mm is not None:
            view = view_level0.to_pixelsize(
                pixelsize_mm=pixelsize_mm, safety_bound=safety_bound,
            )
        elif level is not None:
            view = view_level0.to_level(level)
        else:
            view = view_level0
        # resize annotations
        return view

    def get_view(
        self,
        center=None,
        # level=0, changed
        level=None,
        size_on_level=None,
        location=None,
        size_mm=None,
        pixelsize_mm=None,
        center_mm=None,
        location_mm=None,
        safety_bound=2,
        annotation_id=None,
        margin=0.0,
        margin_in_pixels=False,
    ) -> "View":
        """

        :param location_mm: [horizontal, vertical]
        :param location: [horizontal, vertical]
        :param size_mm: [width, height]
        :param size: [width, height]
        """

        view = View(
            anim=self,
            center=center,
            level=level,
            size_on_level=size_on_level,
            location=location,
            size_mm=size_mm,
            pixelsize_mm=pixelsize_mm,
            center_mm=center_mm,
            location_mm=location_mm,
            safety_bound=safety_bound,
            annotation_id=annotation_id,
            margin=margin,
            margin_in_pixels=margin_in_pixels,
        )
        return view

    def get_views_by_title(
        self, title=None, level=2, return_ids=False, **kwargs
    ) -> List["View"]:
        annotation_ids = self.get_annotation_ids(title)
        if return_ids:
            return self.get_views(annotation_ids, level=level, **kwargs), annotation_ids
        else:
            return self.get_views(annotation_ids, level=level, **kwargs)

    def select_inner_annotations(
        self, id, color=None, raise_exception_if_not_found=False, ann_ids: List = None
    ):
        """

        :param id:
        :param color:
        :param raise_exception_if_not_found:
        :param ann_ids: preselected annotation ids. If None is given (default) all the annotations are considered.
        :return:
        """
        if color is not None:
            an_ids_sel1 = self.get_annotations_by_color(
                color,
                raise_exception_if_not_found=raise_exception_if_not_found,
                ann_ids=ann_ids,
            )
        else:
            if ann_ids is None:
                # an_ids_sel1 = list(self.annotations.keys())
                an_ids_sel1 = list(range(0, len(self.annotations)))
            else:
                an_ids_sel1 = ann_ids

        x_px = self.annotations[id]["x_px"]
        y_px = self.annotations[id]["y_px"]
        ids = []
        for idi in an_ids_sel1:
            ann = self.annotations[idi]
            x_pxi = ann["x_px"]
            y_pxi = ann["y_px"]

            if (
                np.max(x_pxi) < np.max(x_px)
                and np.max(y_pxi) < np.max(y_px)
                and np.min(x_px) < np.min(x_pxi)
                and np.min(y_px) < np.min(y_pxi)
            ):
                ids.append(idi)
        return ids

    def select_outer_annotations(
        self, id, color=None, raise_exception_if_not_found=False, ann_ids: List = None
    ):
        """

        :param id:
        :param color:
        :param raise_exception_if_not_found:
        :param ann_ids: preselected annotation ids. If None is given (default) all the annotations are considered.
        :return:
        """
        if color is not None:
            an_ids_sel1 = self.get_annotations_by_color(
                color,
                raise_exception_if_not_found=raise_exception_if_not_found,
                ann_ids=ann_ids,
            )
        else:
            if ann_ids is None:
                an_ids_sel1 = list(range(0, len(self.annotations)))
            else:
                an_ids_sel1 = ann_ids

        x_px = self.annotations[id]["x_px"]
        y_px = self.annotations[id]["y_px"]
        ids = []
        for idi in an_ids_sel1:
            ann = self.annotations[idi]
            x_pxi = ann["x_px"]
            y_pxi = ann["y_px"]

            if (
                np.max(x_px) < np.max(x_pxi)
                and np.max(y_px) < np.max(y_pxi)
                and np.min(x_pxi) < np.min(x_px)
                and np.min(y_pxi) < np.min(y_px)
            ):
                ids.append(idi)
        return ids

    def select_just_outer_annotations(
        self,
        color,
        return_holes=True,
        ann_ids: List[int] = None,
        raise_exception_if_not_found=True,
    ) -> List:
        """
        Select outer annotation and skip all inner annotations with same color. Put inner annotations to list of holes.
        :param color: set the required annotation color
        :param return_holes: list of inner holes of outer annotations.
        :param ann_ids: put inside pre-filtered list of annotations
        :return:
        """
        ann_ids = self.get_annotations_by_color(
            color,
            ann_ids=ann_ids,
            raise_exception_if_not_found=raise_exception_if_not_found,
        )
        if len(ann_ids) > 0:
            ann_pairs = [
                [aid, self.select_inner_annotations(aid, ann_ids=ann_ids)]
                for aid in ann_ids
                if len(self.select_outer_annotations(aid, ann_ids=ann_ids)) == 0
            ]
            outer_inds, holes = zip(*ann_pairs)
        else:
            outer_inds = []
            holes = []
        if return_holes:
            return list(outer_inds), list(holes)
        else:
            return list(outer_inds)

    def select_annotations_by_title(self, title):
        return self.get_annotation_ids(title)
        # return self.get_views(annotation_ids, level=level, **kwargs), annotation_ids

    def select_annotations_by_title_contains(
        self, title_contains: str
    ) -> annotationIDs:
        # double for
        # return [id for title in self.id_by_titles.keys() if len(re.findall(title_regex, title) > 0 in title for id in self.id_by_titles[title]]
        return [
            id
            for title in self.id_by_titles.keys()
            if title_contains in title
            for id in self.id_by_titles[title]
        ]

    def select_annotations_by_title_regex(self, title_regex: str) -> annotationIDs:
        return [
            id
            for title in self.id_by_titles.keys()
            if len(re.findall(title_regex, title)) > 0
            for id in self.id_by_titles[title]
        ]

    def get_views(
        self,
        annotation_ids=None,
        level=None,
        # margin=0.5,
        margin=0.0,
        margin_in_pixels: bool = False,
        show=False,
        pixelsize_mm=None,
        safety_bound=2,
    ) -> List["View"]:
        """
        Prepare list of possible views according to input parameters.

        :param annotation_ids: list of annotation ids
        :param level: If neither level neither pixelsize_mm is set the level is set to 0
        :param margin: based on "margin_in_pixels" the margin in pixels(accoarding to the requested level) are used or\
        margin is proportional to size of annotation object.
        :param margin_in_pixels: bool
        :param show:
        :return: list of views


        """

        views = [None] * len(annotation_ids)
        for i, annotation_id in enumerate(annotation_ids):
            view = self.get_view(
                annotation_id=annotation_id,
                level=level,
                margin=margin,
                margin_in_pixels=margin_in_pixels,
                pixelsize_mm=pixelsize_mm,
                safety_bound=safety_bound,
            )
            if show:
                view.region_imshow_annotation(annotation_id)
            views[i] = view

        return views

    def set_region(self, center=None, level=0, size=None, location=None):

        if size is None:
            size = [256, 256]

        size = np.asarray(size)

        if location is None:
            location = self.get_region_location_by_center(center, level, size)
        else:
            center = self.get_region_center_by_location(location, level, size)

        self.region_location = location
        self.region_center = center
        self.region_size = size
        self.region_level = level
        self.region_pixelsize, self.region_pixelunit = self.get_pixel_size(level)
        self.level_pixelsize = [
            get_pixelsize(
                self.openslide, level=i, requested_unit=self.region_pixelunit
            )[0]
            for i in range(0, self.openslide.level_count)
        ]
        self._adjust_annotation_to_image_view()

    def _adjust_annotation_to_image_view(self):
        """
        Recalculate annotation from global (level=0) coordinates x_px, y_px to local coordinates
        :return:
        """
        scan.adjust_annotation_to_image_view(
            self.openslide,
            self.annotations,
            self.region_center,
            self.region_level,
            self.region_size,
        )

    def select_annotations_by_color(
        self, id, raise_exception_if_not_found=True, ann_ids=None
    ):
        """

        :param id: number or color '#00FF00
        :param raise_exception_if_not_found:
        :param ann_ids:
        :return:
        """
        logger.warning(
            "Function select_annotations_by_color is deprecated. Use get_annotations_by_color instead."
        )
        self.get_annotations_by_color(
            id,
            raise_exception_if_not_found=raise_exception_if_not_found,
            ann_ids=ann_ids,
        )

    def get_annotations_by_color(
        self, id, raise_exception_if_not_found=True, ann_ids=None
    ):
        """

        :param id: number or color '#00FF00
        :param raise_exception_if_not_found:
        :param ann_ids:
        :return:
        """
        if id is None:
            # probably should return all ids for all colors
            raise ColorError()
            return None

        if type(id) is str:
            if id.upper() not in self.id_by_colors:
                if raise_exception_if_not_found:
                    raise ColorError()
                # return None
                return []
            id = self.id_by_colors[id]
        else:
            id = [id]
        if ann_ids is not None:
            id = [idi for idi in id if id in ann_ids]
        return id

    def get_annotation_ids(
        self, id: Union[annotationIDs, annotationID]
    ) -> annotationIDs:
        if type(id) is str:
            id = self.id_by_titles[id]
        else:
            id = [id]
        return id

    def get_annotation_id(self, i):
        if type(i) is str:
            i = self.id_by_titles[i][0]
        return i

    def set_region_on_annotations(self, i=None, level=2, boundary_px=10, show=False):
        """

        :param i: index of annotation or annotation title
        :param level:
        :param boundary_px:
        :return:
        """
        i = self.get_annotation_id(i)
        center, size = self.get_annotations_bounds_px(i)
        region_size = (
            (size / self.openslide.level_downsamples[level]) + 2 * boundary_px
        ).astype(int)
        self.set_region(center=center, level=level, size=region_size)
        if show:
            self.region_imshow_annotation(i)

    def get_annotations_bounds_px(self, i=None):
        """
        Bounds are in pixels on level 0.
        :param i:
        :return:
        """
        i = self.get_annotation_id(i)
        if i is not None:
            anns = [self.annotations[i]]

        x_px = []
        y_px = []

        for ann in anns:
            x_px.append(ann["x_px"])
            y_px.append(ann["y_px"])

        mx = np.array([np.max(x_px), np.max(y_px)])
        mi = np.array([np.min(x_px), np.min(y_px)])
        all = [mi, mx]
        center = np.mean(all, 0)
        size = mx - mi
        return center, size

    def get_region_image(self, as_gray=False, as_unit8=False):

        location = fix_location_ndpi(
            self.openslide, self.region_location, self.region_level
        )
        imcr = self.openslide.read_region(
            location, level=self.region_level, size=self.region_size
        )
        im = np.asarray(imcr)
        if as_gray:
            if len(im.shape) > 2:
                if im.shape[2] == 4:
                    im = skimage.color.rgba2rgb(im)
                im = skimage.color.rgb2gray(im)
            if as_unit8:
                im = (im * 255).astype(np.uint8)

        im = self._run_raster_image_preprocessing_function_handler(im)
        if self.run_intensity_rescale:
            im = self.intensity_rescaler.rescale_intensity(im)
        return im

    def plot_annotations(self, annotation_id=None, fontsize="x-small"):
        if annotation_id is None:
            anns = self.annotations
        else:
            annotation_id = self.get_annotation_id(annotation_id)
            anns = [self.annotations[annotation_id]]
        scan.plot_annotations(anns, in_region=True, fontsize=fontsize)

    def get_annotation_region_raster(self, i):
        i = self.get_annotation_id(i)
        polygon_x = self.annotations[i]["region_x_px"]
        polygon_y = self.annotations[i]["region_y_px"]
        polygon = list(zip(polygon_y, polygon_x))
        poly_path = mplPath(polygon)

        x, y = np.mgrid[: self.region_size[1], : self.region_size[0]]
        # x, y = np.mgrid[: self.region_size[0], : self.region_size[1]] # TODO swap
        coors = np.hstack(
            (x.reshape(-1, 1), y.reshape(-1, 1))
        )  # coors.shape is (4000000,2)

        mask = poly_path.contains_points(coors)
        mask = mask.reshape(self.region_size[::-1])
        # mask = mask.reshape(self.region_size) # TODO swap
        return mask

    def region_imshow_annotation(self, i=None):
        region = self.get_region_image()
        plt.imshow(region)
        self.plot_annotations(i)

    def coords_region_px_to_global_px(self, points_view_px: np.ndarray):
        """
        :param points_view_px: np.asarray([[x0, x1, ...], [y0, y1, ...]])
        :return:
        """
        px_factor = self.openslide.level_downsamples[self.region_level]
        x_px = self.region_location[0] + points_view_px[0] * px_factor
        y_px = self.region_location[1] + points_view_px[1] * px_factor

        return x_px, y_px

    def coords_global_px_to_view_px(self, points_glob_px: np.ndarray):
        """
        :param points_glob_px: np.asarray([[x0, x1, ...], [y0, y1, ...]])
        :return:
        """

        px_factor = self.openslide.level_downsamples[self.region_level]
        x_glob_px = points_glob_px[0]
        y_glob_px = points_glob_px[1]
        x_view_px = (x_glob_px - self.region_location[0]) / px_factor
        y_view_px = (y_glob_px - self.region_location[1]) / px_factor

        return x_view_px, y_view_px


class View:
    def __init__(
        self,
        anim: AnnotatedImage,
        center=None,
        level: int = 0,
        size_on_level=None,
        location=None,
        size_mm=None,
        pixelsize_mm=None,
        center_mm=None,
        location_mm=None,
        safety_bound=2,
        annotation_id=None,
        margin=0.0,
        margin_in_pixels: bool = False,
    ):
        """
        Get view on the image defined by 3 numbers like center, level and size.


        :param location_mm: [horizontal, vertical]
        :param location: [horizontal, vertical]
        :param size_mm: [width, height]
        :param size: [width, height]

        """
        self.anim: AnnotatedImage = anim
        self._requested_size_on_level_when_defined_by_pixelsize = None
        self._is_resized_by_pixelsize = None

        self.set_region(
            center=center,
            level=level,
            size_on_level=size_on_level,
            location=location,
            size_mm=size_mm,
            pixelsize_mm=pixelsize_mm,
            center_mm=center_mm,
            location_mm=location_mm,
            safety_bound=safety_bound,
            annotation_id=annotation_id,
            margin=margin,
            margin_in_pixels=margin_in_pixels,
        )
        self.select_outer_annotations = self.anim.select_outer_annotations
        self.select_inner_annotations = self.anim.select_inner_annotations
        self.get_raster_image = self.get_region_image
        self.get_annotation_region_raster = self.get_annotation_raster

    def set_region(
        self,
        center=None,
        level: int = None,
        size_on_level=None,
        location=None,
        size_mm=None,
        pixelsize_mm=None,
        center_mm=None,
        location_mm=None,
        safety_bound=2,
        annotation_id=None,
        # margin=0.5,
        margin=0.0,
        margin_in_pixels: bool = False,
    ):
        """

        :param location_mm: [horizontal, vertical]
        :param location: [horizontal, vertical]
        :param size_mm: [width, height]
        :param size: [width, height]
        """
        self._requested_size_on_level_when_defined_by_pixelsize = None
        if (level is None) and (pixelsize_mm is None) and (size_on_level is not None):
            raise ValueError(
                "Parameter 'size_on_level' cannot be used. Define 'level' or 'pixelsize_mm' to fix it."
            )
        if pixelsize_mm is not None:
            self._is_resized_by_pixelsize = True
            if np.isscalar(pixelsize_mm):
                pixelsize_mm = [pixelsize_mm, pixelsize_mm]
            pixelsize_mm = np.asarray(pixelsize_mm)
            self.region_pixelsize = pixelsize_mm
            self.region_pixelunit = "mm"
            if level is None:
                level = self.anim.get_optimal_level_for_fluent_resize(
                    self.region_pixelsize, safety_bound=safety_bound
                )
                if size_on_level is not None:
                    _region_pixelsize, _region_pixelunit = self.get_pixelsize_on_level(
                        level=level
                    )
                    alpha = self.region_pixelsize / _region_pixelsize
                    self._requested_size_on_level_when_defined_by_pixelsize = (
                        size_on_level
                    )
                    size_on_level = size_on_level * alpha
                    size_on_level = size_on_level.astype(int)

        else:
            if level is None:
                level = 0
            self._is_resized_by_pixelsize = False
            # self.region_pixelsize = None
            # self.region_pixelunit = "mm"
            self.region_pixelsize, self.region_pixelunit = self.get_pixelsize_on_level(
                level
            )
        if annotation_id is not None:
            center, size_on_level0 = self.anim.get_annotations_bounds_px(annotation_id)
            if size_on_level is None and size_mm is None:
                size_on_level = (
                    size_on_level0 / self.anim.openslide.level_downsamples[level]
                ).astype(int)

        if size_mm is not None:
            if np.isscalar(size_mm):
                size_mm = [size_mm, size_mm]
            if size_on_level is not None:
                raise ValueError("Parameter size and size_mm are exclusive.")
            size_mm = np.asarray(size_mm)
            size_on_level = np.ceil(
                size_mm / self.get_pixelsize_on_level(level)[0]
            ).astype(int)

        if size_on_level is None:
            size_on_level = [256, 256]

        size_on_level = np.asarray(size_on_level)

        ## every size is now converted to size_on_level
        size_on_level, margin_px_on_level = self._calculate_size_based_on_margin(
            size_on_level, margin, margin_in_pixels
        )

        self.region_level = level
        self.region_size_on_level = size_on_level
        self.zoom: np.ndarray  # additional zoom to size on level

        if pixelsize_mm is not None:
            pxsz = self.get_pixelsize_on_level(level)[0]
            self.zoom = pxsz / (1.0 * pixelsize_mm)
            self.region_size_on_pixelsize_mm = np.ceil(
                size_on_level * self.zoom
            ).astype(int)
        else:
            self.region_size_on_pixelsize_mm = size_on_level
            self.zoom = np.array([1, 1])

        if location_mm is not None:
            location = (location_mm / self.get_pixelsize_on_level(0)[0]).astype(int)
        if center_mm is not None:
            center = (np.asarray(center_mm) / self.get_pixelsize_on_level(0)[0]).astype(
                int
            )

        # now we have location and center (all _mm variants are not used any more)
        # if there is some margin, convert to center view
        if (center is None) and (margin_px_on_level != 0).any():
            location = location - margin_px_on_level
            # center = self.get_region_center_by_location(location, level, size_on_level)

        if location is None:
            location = self.get_region_location_by_center(center, level, size_on_level)
        else:
            if center is not None:
                logger.error("Can take location or center. Not both of them.")
                return
            center = self.get_region_center_by_location(location, level, size_on_level)

        self.region_location = location
        self.region_center = center

        self.set_annotations(self.anim.annotations)

        # scan.adjust_annotation_to_image_view(
        #     self.anim.openslide, self.annotations, center, level, size_on_level
        # )

    def __str__(self):
        return f"View: location={self.region_location}, level={self.region_level}, size_on_level={self.region_size_on_level}"

    def set_annotations(self, annotations):
        import copy

        self.annotations = copy.deepcopy(annotations)
        self.adjust_annotation_to_image_view()

    def adjust_annotation_to_image_view(self):
        """
        Recalculate annotation from global (level=0) coordinates x_px, y_px to local coordinates
        :return:
        """
        scan.adjust_annotation_to_image_view(
            self.anim.openslide,
            self.annotations,
            self.region_center,
            self.region_level,
            self.region_size_on_level,
        )

    def _get_margin_px(self, size_on_level, margin, margin_in_pixels: bool):
        if margin_in_pixels:
            margin_px = int(margin)
        else:
            margin_px = (size_on_level * margin).astype(
                int
            )  # / self.anim.openslide.level_downsamples[level]
        return margin_px

    def _calculate_size_based_on_margin(
        self, size_on_level, margin, margin_in_pixels: bool
    ):
        margin_px_on_level = self._get_margin_px(
            size_on_level, margin, margin_in_pixels
        )
        # if (size_on_level is None) and (size_mm is None):
        # if (size_on_level is None) and (size_mm is None):
        size_on_level = (
            size_on_level
            # (size_on_level0 / self.anim.openslide.level_downsamples[level])
            + 2 * margin_px_on_level
        ).astype(int)
        return size_on_level, margin_px_on_level

    def get_pixelsize_on_level(self, level=None):
        if level is None:
            level = self.region_level
        return self.anim.get_pixel_size(level)

    def mm_to_px(self, mm):
        pxsz = self.region_pixelsize
        # pxsz, unit = self.get_pixelsize_on_level(level=level)
        # if unit is not "mm":
        #     raise Exception(f"Wrong unit. Expected 'mm', given '{unit}'.")
        return mm / pxsz

    def region_imshow_annotation(self, i):
        region = self.get_region_image()
        plt.imshow(region)
        self.plot_annotations(i)

    def coords_glob_px_to_view_px(self, x_glob_px, y_glob_px):
        # px_factor = self.anim.openslide.level_downsamples[self.region_level]
        px_factor = self.region_pixelsize / self.get_pixelsize_on_level(0)[0]

        x_px = (x_glob_px - self.region_location[0]) / px_factor[0]
        y_px = (y_glob_px - self.region_location[1]) / px_factor[1]

        return x_px, y_px

    def coords_view_px_to_glob_px(self, x_view_px, y_view_px):
        """
        :param x_view_px: [x0, x1, ...]
        :param y_view_px: [y0, y1, ...]]
        :return:
        """
        px_factor = self.region_pixelsize / self.get_pixelsize_on_level(0)[0]
        # px_factor = self.anim.openslide.level_downsamples[self.region_level]
        # print(px_factor)
        x_px = self.region_location[0] + x_view_px * px_factor[0]
        y_px = self.region_location[1] + y_view_px * px_factor[1]

        return x_px, y_px

    def plot_points(self, x_glob_px, y_glob_px):
        # points = [x_glob_px, y_glob_px]
        x_view_px, y_view_px = self.coords_glob_px_to_view_px(x_glob_px, y_glob_px)
        plt.plot(x_view_px, y_view_px, "oy")

    def get_annotation_raster(
        self, annotation_id: int, holes_ids: List[int] = None
    ) -> np.ndarray:
        if holes_ids is None:
            holes_ids = []
        ann_raster = self._get_single_annotation_region_raster(annotation_id)
        # if len(holes_ids) == 0:
        #     # ann_raster = ann_raster1
        # else:
        for hole_id in holes_ids:

            ann_raster2 = self.get_annotation_raster(hole_id)
            ann_raster = ann_raster ^ ann_raster2
        return ann_raster

    def get_annotation_raster_by_color(
        self, color, make_holes=True, raise_exception_if_not_found: bool = False
    ):
        """
        Prepare raster image with all annotation with the defined color
        :param color: requested annotation color
        :param make_holes: make whole if the same annotation label is inside
        :param raise_exception_if_not_found: control the exception. It is raised if the color is not found.
        :return:
        """
        outer_ids, holes_ids = self.anim.select_just_outer_annotations(
            color=color, raise_exception_if_not_found=raise_exception_if_not_found
        )
        # segmentation_one_color = None
        # polygon_x = self.annotations[annotation_id]["region_x_px"] * self.zoom[0]
        # polygon_y = self.annotations[annotation_id]["region_y_px"] * self.zoom[1]
        # TODO swap axes
        segmentation_one_color = np.zeros(
            [self.region_size_on_pixelsize_mm[1], self.region_size_on_pixelsize_mm[0]],
            dtype=np.uint8,
        )
        for outer_id, hole_ids in zip(outer_ids, holes_ids):
            if make_holes:
                ann_raster = self.get_annotation_raster(outer_id, holes_ids=hole_ids)
            else:
                ann_raster = self.get_annotation_raster(outer_id)
            # if segmentation_one_color is None:
            #     segmentation_one_color = ann_raster
            segmentation_one_color += ann_raster
        return segmentation_one_color

    def _get_single_annotation_region_raster(self, annotation_id):
        annotation_id = self.anim.get_annotation_id(annotation_id)
        # Coordinates swap
        # coordinates are swapped here. Probably it is because Path uses different order from Image
        polygon_x = self.annotations[annotation_id]["region_x_px"] * self.zoom[0]
        polygon_y = self.annotations[annotation_id]["region_y_px"] * self.zoom[1]
        polygon = list(zip(polygon_y, polygon_x))
        poly_path = mplPath(polygon)

        # coordinates are swapped also here TODO
        # x, y = np.mgrid[: self.region_size_on_level[1], : self.region_size_on_level[0]]
        x, y = np.mgrid[
            : self.region_size_on_pixelsize_mm[1], : self.region_size_on_pixelsize_mm[0]
        ]
        coors = np.hstack(
            (x.reshape(-1, 1), y.reshape(-1, 1))
        )  # coors.shape is (4000000,2)

        mask = poly_path.contains_points(coors)
        # swap TODO
        mask = mask.reshape(self.region_size_on_pixelsize_mm[::-1])
        return mask

    def region_imshow_annotation(self, i=None):
        region = self.get_region_image()
        plt.imshow(region)
        self.plot_annotations(i)
        self.add_ticks()

    def add_ticks(self, print_units: bool = True, format: str = "{:.1e}"):
        self.region_pixelunit
        region_pixelsize = self.region_pixelsize
        locs, labels = plt.xticks()
        labels = [format.format(i * region_pixelsize[0]) for i in locs]
        plt.xticks(locs[1:-1], labels[1:-1], rotation="vertical")

        locs, labels = plt.yticks()
        labels = [format.format(i * region_pixelsize[1]) for i in locs]
        plt.yticks(locs[1:-1], labels[1:-1])
        ax = plt.gca()
        # ax.text()

        if print_units:
            plt.ylabel(str(self.region_pixelunit))
            # plt.text(-0.1, -0.1, str(self.region_pixelunit) + "asdf")

    def plot_annotations(self, i=None, fontsize="x-small"):
        if i is None:
            anns = self.annotations
        else:
            i = self.anim.get_annotation_id(i)
            anns = [self.annotations[i]]
        scan.plot_annotations(anns, in_region=True, factor=self.zoom, fontsize=fontsize)

    def get_region_location_by_center(self, center, level, size):
        return get_region_location_by_center(self.anim.openslide, center, level, size)

    def get_region_center_by_location(self, location, level, size):
        return get_region_center_by_location(self.anim.openslide, location, level, size)

    def get_region_image_resolution(
        self, resolution_mm, as_gray=False,
    ):
        self.anim.openslide.level_downsamples

    def get_region_image(self, as_gray=False, log_level="TRACE"):
        """
        Get raster image from the view. It can have defined pixelsize, and also the level
        :param as_gray:
        :param pixelsize_mm:
        :param safety_bound: Resize safety multiplicator. If set to 2 it means to have at least 2 samples per pixel
        along each axis.
        :param level: The level of output image can be controled by this parameter. The computation of optimal level
        can be skipped by this.

        :return:
        """

        location = self.region_location
        if self.anim.image_type == ".ndpi":
            location = fix_location_ndpi(
                self.anim.openslide, location, self.region_level
            )
        imcr = self.anim.openslide.read_region(
            # TODO here should be changed order of size
            location, level=self.region_level, size=self.region_size_on_level
        )
        im = np.asarray(imcr)

        if len(im.shape) > 2:
            im4stat = im[:, :, :3]
        else:
            im4stat = im[:, :]
        logger.log(
            log_level,
            f"imcr dtype: {im.dtype}, shape: {im.shape}, min max: [{np.min(im4stat)}, {np.max(im4stat)}]",
        )
        # logger.debug(f"imcr dtype: {im.dtype}, shape: {im.shape}, min max: [{np.min(im[:,:,:3])}, {np.max(im[:,:,:3])}], mean: {np.mean(im[:,:,:3])}, min max alpha: [{np.min(im[:,:,3])}, {np.max(im[:,:,3])}], mean: {np.mean(im[:,:,3])}")
        logger.log(log_level, "Do intensity rescale if necessary")
        im = self.anim._run_raster_image_preprocessing_function_handler(im)
        if self.anim.run_intensity_rescale:
            im = self.anim.intensity_rescaler.rescale_intensity(im)

        if as_gray:
            if len(im.shape) > 2:
                if im.shape[2] == 4:
                    logger.log(log_level, "RGBA to RGB...")
                    # im = skimage.color.rgba2rgb(im)
                    im = im[:, :, :3]
                logger.log(log_level, "RGB to gray ...")
                im = skimage.color.rgb2gray(im)

        logger.log(log_level, "resize if resized by pixelsize")
        if self._is_resized_by_pixelsize:
            logger.log(log_level, "Resized by pixelsize")
            pxsz_level, pxunit_level = self.anim.get_pixel_size(level=self.region_level)
            # swap coordinates because openslice output image have swapped image coordinates
            im_resized = imma.image.resize_to_mm(
                im, pxsz_level[::-1], self.region_pixelsize[::-1], anti_aliasing=True
            )
            logger.log(log_level, f"pxsz_level={pxsz_level}")
            req_sz = self._requested_size_on_level_when_defined_by_pixelsize
            if req_sz is not None:
                req_sz = np.asarray(req_sz)
                if not np.array_equal(im_resized.shape[:2], req_sz):
                    # Array should be the same size.
                    # Due to numerical error in alpha computation there can be small pixel error
                    norm = np.linalg.norm(im_resized.shape[:2] - req_sz)
                    if norm > 3.0:
                        logger.error(
                            f"Requested size ({req_sz}) differ "
                            f"from the real image size ({im_resized.shape}) a lot. Fixing by resize."
                        )
                    # not sure about [::-1]. Not checked too much.
                    im_resized = imma.image.resize_to_shape(im, req_sz[::-1])
            im = im_resized

        return im

    def imshow(self, as_gray=False):
        plt.imshow(self.get_region_image(as_gray=as_gray))

    def get_size_on_level(self, new_level):
        imsl = self.anim.openslide
        former_level = self.region_level
        former_size = self.region_size_on_level
        scale_factor = (
            imsl.level_downsamples[former_level] / imsl.level_downsamples[new_level]
        )
        new_size = (np.asarray(former_size) * scale_factor).astype(int)

        return new_size

    def get_size_on_pixelsize_mm(self):
        return self.region_size_on_pixelsize_mm

    def to_level(self, new_level):
        size = self.get_size_on_level(new_level)
        newview = View(
            self.anim,
            location=self.region_location,
            size_on_level=size,
            level=new_level,
        )
        return newview

    def to_pixelsize(self, pixelsize_mm, safety_bound=2.0):
        level = self.anim.get_optimal_level_for_fluent_resize(
            pixelsize_mm, safety_bound=safety_bound
        )
        size = self.get_size_on_level(level)
        newview = View(
            self.anim,
            location=self.region_location,
            size_on_level=size,
            level=level,
            pixelsize_mm=pixelsize_mm,
        )
        return newview

    def get_slices_for_insert_image_from_view(self, other_view: "View"):
        """
        Prepare region slice for inserting small image into current view.
        :param other_view:
        :return:
        """
        pixelsize_factor = self.get_pixelsize_on_level(0)[0] / self.region_pixelsize
        delta_px = (
            (other_view.region_location - self.region_location) * pixelsize_factor
        ).astype(int)
        # start = (self.region_location + delta_px)[::-1]
        # delta_size_px = (other_view.get_size_on_pixelsize_mm() * pixelsize_factor).astype(int)
        start = delta_px[::-1]
        import imma.image

        pxsz1 = self.region_pixelsize[:2]
        pxsz2 = other_view.region_pixelsize[:2]
        stop = start + imma.image.calculate_new_shape(
            other_view.get_size_on_pixelsize_mm()[::-1],
            voxelsize_mm=pxsz2[::-1],
            new_voxelsize_mm=pxsz1[::-1],
        )
        return (slice(start[0], stop[0], 1), slice(start[1], stop[1], 1))

    def insert_image_from_view(self, other_view: "View", img, other_img):
        """
        Put small image from different view into current image. Current image should be bigger.

        :param other_view:
        :param img:
        :param other_img:
        :return:
        """
        import copy

        img_copy = copy.copy(img)

        pxsz1 = self.region_pixelsize[:2]
        pxsz2 = other_view.region_pixelsize[:2]
        resized_other_img = imma.image.resize_to_mm(
            other_img,
            voxelsize_mm=pxsz2[::-1],
            new_voxelsize_mm=pxsz1[::-1],
            anti_aliasing=True,
        )

        sl = self.get_slices_for_insert_image_from_view(other_view)

        img_copy[sl] = resized_other_img
        return img_copy

    def get_full_view(
        self,
        level=None,
        pixelsize_mm=None,
        safety_bound=2,
        margin=0.0,
        margin_in_pixels=False,
        # annotation_id=None,
        # margin=0.5,
        # margin_in_pixels=False,
    ) -> "View":
        self.anim.get_full_view(
            level=level,
            pixelsize_mm=pixelsize_mm,
            safety_bound=safety_bound,
            margin=margin,
            margin_in_pixels=margin_in_pixels,
        )

    def get_training_labels(self, fill_gaps=False, return_debug_images=False):
        view = self
        seg_black = view.get_annotation_raster_by_color(
            "#000000", raise_exception_if_not_found=False
        )
        seg_magenta = view.get_annotation_raster_by_color(
            "#FF00FF", raise_exception_if_not_found=False
        )
        seg_red = view.get_annotation_raster_by_color(
            "#FF0000", raise_exception_if_not_found=False
        )
        # find overlays
        overlays = (1 * seg_black + 1 * seg_magenta + 1 * seg_red) > 1
        segmentation = 2 * seg_black + 1 * seg_magenta + 3 * seg_red
        # remove overlays
        segmentation[overlays] = 0
        if not fill_gaps:
            if return_debug_images:
                return segmentation, []
            return segmentation
        else:
            from scipy.ndimage import morphology

            dst, inds = morphology.distance_transform_edt(
                segmentation == 0, return_indices=True
            )
            # plt.imshow(dst)

            # fill the gaps
            filled = segmentation[[*inds]]
            # plt.imshow(filled, vim=1)
            # plt.colorbar()
            if return_debug_images:
                return filled, [dst]
            else:
                return filled


class ColorError(Exception):
    pass


def imshow_with_colorbar(*args, **kwargs):
    ax = plt.gca()
    im = ax.imshow(*args, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)


def get_offset_px(imsl: ImageSlide):

    pm = imsl.properties
    pixelsize, pixelunit = get_pixelsize(imsl, requested_unit="mm")
    offset = np.asarray(
        (
            int(pm["hamamatsu.XOffsetFromSlideCentre"]),
            int(pm["hamamatsu.YOffsetFromSlideCentre"]),
        )
    )
    # resolution_unit = pm["tiff.ResolutionUnit"]
    offset_mm = offset * 0.000001
    if pixelunit is not "mm":
        raise ValueError(f"Cannot convert pixelunit {pixelunit} to milimeters")
    offset_from_center_px = offset_mm / pixelsize
    im_center_px = np.asarray(imsl.dimensions) / 2.0
    offset_px = im_center_px - offset_from_center_px
    return offset_px
