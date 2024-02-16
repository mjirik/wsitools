# /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modul is used for GUI of Lisa
"""
from loguru import logger

# problem is loading lxml together with openslide
# from lxml import etree
import os
import json
import os.path as op
import glob
import matplotlib.pyplot as plt
import numpy as np
import re
import math
from read_roi import read_roi_zip

__version__ = "0.34.2"

# def get_one_annotation(viewstate):
#     titles_list = viewstate.xpath(".//title/text()")
#     if len(titles_list) == 0:
#         an_title = ""
#     elif len(titles_list) == 1:
#         an_title = titles_list[0]
#     else:
#         raise ValueError("More than one title in viewstate")
#     details_list = viewstate.xpath(".//details/text()")
#     if len(details_list) == 0:
#         an_details = ""
#     elif len(details_list) == 1:
#         an_details = details_list[0]
#     else:
#         raise ValueError("More than one details in viewstate")
#
#     annotations = viewstate.xpath(".//annotation")
#     if len(annotations) > 1:
#         raise ValueError("More than one annotation found")
#     annot = annotations[0]
#     an_color = annot.get("color")
#     #     display(len(annotation))
#     an_x = list(map(int, annot.xpath(".//pointlist/point/x/text()")))
#     an_y = list(map(int, annot.xpath(".//pointlist/point/y/text()")))
#     return dict(title=an_title, color=an_color, x=an_x, y=an_y, details=an_details)


def get_one_annotation(viewstate):
    def get_text(el):
        tx = el.text
        return "" if tx is None else tx

    titles_list = viewstate.findall(".//title")
    if len(titles_list) == 0:
        an_title = ""
    elif len(titles_list) == 1:
        an_title = get_text(titles_list[0])
    else:
        raise ValueError("More than one title in viewstate")
    details_list = viewstate.findall(".//details")
    if len(details_list) == 0:
        an_details = ""
    elif len(details_list) == 1:
        an_details = get_text(details_list[0])
    else:
        raise ValueError("More than one details in viewstate")

    annotations = viewstate.findall(".//annotation")
    if len(annotations) > 1:
        raise ValueError("More than one annotation found")
    annot = annotations[0]
    an_color = annot.get("color")
    #     display(len(annotation))
    an_x = list(map(int, map(get_text, annot.findall(".//pointlist/point/x"))))
    an_y = list(map(int, map(get_text, annot.findall(".//pointlist/point/y"))))
    return dict(title=an_title, color=an_color, x=an_x, y=an_y, details=an_details)


# def _ndpa_file_to_json(pth):
#
#     # problem is loading lxml together with openslide
#     from lxml import etree
#
#     tree = etree.parse(pth)
#     viewstates = tree.xpath("//ndpviewstate")
#     all_anotations = list(map(get_one_annotation, viewstates))
#     fn = pth + ".json"
#     with open(fn, "w") as outfile:
#         json.dump(all_anotations, outfile)


def _ndpa_file_to_json(pth):
    # problem is loading lxml together with openslide
    # from lxml import etree

    # with xml there is no need to use subprocess anymore
    import xml.etree.ElementTree as etree

    tree = etree.parse(pth)
    viewstates = tree.findall("//ndpviewstate")
    all_anotations = list(map(get_one_annotation, viewstates))
    fn = pth + ".json"
    with open(fn, "w") as outfile:
        json.dump(all_anotations, outfile)


def ndpa_to_json(path):
    """
    :param path: path to file or dir contaning .ndpa files
    """
    # print(os.getenv("PATH"))
    syspth = str(os.getenv("PATH"))
    ind = syspth.find("openslide")
    st = max(0, ind - 30)
    sp = min(len(syspth), ind + 30)
    if ind < 0:
        logger.debug(f"Not found 'openslide' in PATH: {syspth}")
    else:
        logger.debug(f"PATH: ...{syspth[st:sp]}...")
    if op.isfile(path):
        fn, ext = op.splitext(path)
        if ext == ".ndpi":
            path = path + ".ndpa"
        if op.exists(path):
            _ndpa_file_to_json(path)
        else:
            logger.info(f"No annotation file found '{path}'")
    else:
        extended_path = op.join(path, "*.ndpa")
        #         print(extended_path)
        files = glob.glob(extended_path)
        for fl in files:
            _ndpa_file_to_json(fl)


def get_imsize_from_imagej_roi(rois):
    """
    Get size from ImageJ annotation. There have to be rectangle over the whole image.
    :param rois:
    :return:
    """
    width = 0
    height = 0
    rect = False
    for roikey in rois:
        roi = rois[roikey]
        #         print(roi["type"])
        if roi["type"] == "rectangle":
            widthi = roi["width"]
            heighti = roi["height"]
            if widthi > width:
                width = widthi
            if heighti > height:
                height = heighti

            rect = True
    if not rect:
        raise Exception("There should be rectangle in ROI file to define image size.")
    return np.array([height, width])


def read_annotations_imagej(path, slide_size) -> list:
    """
    Read annotation from ImageJ. It is stored in `.roi.zip` file. There should be one rectangle annotation over
    whole image to allow get size of the image used to produce the ROI.
    All polygon-type annotations should contain color code in name like #FF0000 for red.
    :param path: Path to original image file. The ROI file name is derived from by adding .roi.zip
    :param slide_size:
    :return:
    """
    # import io3d
    # import numpy as np
    fn = path + ".roi.zip"
    logger.debug(f"Looking for ROI file {fn}")
    anns = []
    if op.exists(fn):
        # def read_annotations_imagej(, slide_size=slide_size):
        rois = read_roi_zip(fn)
        roi_size = get_imsize_from_imagej_roi(rois)
        ratio = np.asarray(slide_size) / roi_size
        if not math.isclose(ratio[0], ratio[1], rel_tol=0.01):
            logger.warning(
                f"ROI size ratio is different from image data. Image size={slide_size}, ROI size={roi_size}"
            )
        logger.debug(f"ratio={ratio}")
        for roi_key in rois:
            one = rois[roi_key]
            if one["type"] == "polygon":
                an_title = one["name"]
                m = re.search(r"#[0-9a-fA-F]{6}", one["name"])
                if m is None:
                    an_color = "#FFFFFF"
                else:
                    an_color = m.group(0)
                # swap
                an_x = np.asarray(one["x"]) * ratio[0]
                an_y = np.asarray(one["y"]) * ratio[1]
                an_details = ""
                one_ann = dict(
                    title=an_title,
                    color=an_color,
                    x_px=an_x,
                    y_px=an_y,
                    details=an_details,
                )
                anns.append(one_ann)
    return anns


def read_annotations_ndpa(pth) -> list:
    """
    Read the ndpa annotations. Annotation is converted to json if it is not done before. This step
    works on Linux but not on Windows.
    :param pth: path to .ndpi file
    :return: readed annotatios
    """

    import platform

    if platform.system() == "Windows":
        import subprocess
        import sys

        # output = subprocess.check_output(["pwd"])
        # print(output)
        # output = subprocess.check_output(["where", "python"])
        # print(output)

        cwd = op.dirname(op.dirname(__file__))
        command = [sys.executable, "-m", "scaffan.ann_to_json", pth]
        try:
            output = subprocess.check_output(command, cwd=cwd, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.debug(f"Command {' '.join(command)}")
            logger.debug(f"Command '{e.cmd}' returned with code {e.returncode}")
            logger.debug(f"Output of command 1: \n{str(e.output)}")
            # logger.debug(f"Output of command 2: \n{e.output.decode()}")
            exit(e.returncode)
            # raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))

        logger.debug("windows annotation output:" + str(output))
        # print(output)
    else:
        # if not op.exists(fn):
        ndpa_to_json(pth)

    fn = pth + ".ndpa.json"
    if op.exists(fn):
        with open(fn) as f:
            data = json.load(f)
    else:
        data = []
    return data


def plot_annotations(
    annotations,
    x_key="x",
    y_key="y",
    in_region=False,
    factor=[1, 1],
    show_id=True,
    fontsize="x-small",
):
    if type(annotations) is dict:
        annotations = [annotations]

    if in_region:
        x_key = "region_x_px"
        y_key = "region_y_px"

    for i, annotation in enumerate(annotations):
        x = np.asarray(annotation[x_key]) * factor[0]
        y = np.asarray(annotation[y_key]) * factor[1]
        # plt.hold(True)
        if len(x) == 0 or len(y) == 0:
            logger.debug(f"Annotation id={i} has zero length")
        else:
            plt.plot(x, y, c=annotation["color"])
            if show_id:
                plt.text(
                    np.min(x),
                    np.min(y),
                    str(i),
                    c=annotation["color"],
                    fontsize=fontsize,
                )


def adjust_xy_to_image_view(imsl, x_px, y_px, center, level, size):
    x_px_view = ((x_px - center[0]) / imsl.level_downsamples[level]) + (size[0] / 2)
    y_px_view = ((y_px - center[1]) / imsl.level_downsamples[level]) + (size[1] / 2)
    return x_px_view, y_px_view


def adjust_annotation_to_image_view(imsl, annotations, center, level, size):
    output = []
    for annotation in annotations:
        ann_out = annotation
        x_px_view, y_px_view = adjust_xy_to_image_view(
            imsl, annotation["x_px"], annotation["y_px"], center, level, size
        )
        ann_out["region_x_px"] = x_px_view
        ann_out["region_y_px"] = y_px_view
        ann_out["region_center"] = center
        ann_out["region_level"] = level
        ann_out["region_size"] = size
        output.append(ann_out)

    return output


def annotation_px_to_mm(imsl: "openslide.OpenSlide", annotation: dict) -> dict:
    """
    Calculate x,y in mm from xy in pixels
    :param imsl:
    :param annotation:
    :return:
    """
    imsl.level_downsamples
    from scaffan.image import get_offset_px, get_pixelsize

    offset_px = get_offset_px(imsl)
    pixelsize, pixelunit = get_pixelsize(imsl, requested_unit="mm")
    x_px = np.asarray(annotation["x_px"])
    y_px = np.asarray(annotation["y_px"])
    x_mm = pixelsize[0] * (x_px - offset_px[0])
    y_mm = pixelsize[1] * (y_px - offset_px[1])
    annotation["x_mm"] = x_mm
    annotation["y_mm"] = y_mm
    return annotation


def annotations_px_to_mm(imsl, annotations):
    """
    Add x_mm and y_mm based on x_px and y_px into annotation list
    :param imsl:
    :param annotations:
    :return:
    """
    from scaffan.image import get_offset_px, get_pixelsize

    offset_px = get_offset_px(imsl)
    pixelsize, pixelunit = get_pixelsize(imsl, requested_unit="mm")
    for annotation in annotations:
        annotation_px_to_mm(imsl, annotation)

    return annotations


def annotations_to_px(imsl, annotations):
    from scaffan.image import get_offset_px, get_pixelsize

    offset_px = get_offset_px(imsl)
    pixelsize, pixelunit = get_pixelsize(imsl, requested_unit="mm")
    for annotation in annotations:
        x_nm = np.asarray(annotation["x"])
        y_nm = np.asarray(annotation["y"])
        x_mm = x_nm * 0.000001
        y_mm = y_nm * 0.000001
        x_px = x_mm / pixelsize[0] + offset_px[0]
        y_px = y_mm / pixelsize[1] + offset_px[1]
        annotation["x_nm"] = x_nm
        annotation["y_nm"] = y_nm
        annotation["x_mm"] = x_mm
        annotation["y_mm"] = y_mm
        annotation["x_px"] = x_px
        annotation["y_px"] = y_px
    return annotations


def annotation_titles(annotations):
    titles = {}
    for i, an in enumerate(annotations):
        title = an["title"]
        if title in titles:
            titles[title].append(i)
        else:
            titles[title] = [i]

    return titles


def annotation_colors(annotations):
    # titles = {}
    colors = {}
    for i, an in enumerate(annotations):
        title = an["color"]
        title = title.upper()
        if title in colors:
            colors[title].append(i)
        else:
            colors[title] = [i]

    return colors


def annotation_details(annotations):
    return _get_annotation_elements(annotations, "details")


def _get_annotation_elements(annotations, element_keyword):
    colors = {}
    for i, an in enumerate(annotations):
        title = an[element_keyword]
        title = title.upper()
        if title in colors:
            colors[title].append(i)
        else:
            colors[title] = [i]

    return colors
