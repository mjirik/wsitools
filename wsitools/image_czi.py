# /usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""

from loguru import logger
import numpy as np
import skimage.transform
import sys
import subprocess
import time


def instal_codecs_with_pip():
    try:
        import imagecodecs
    except ImportError as e:
        logger.info("Installing imagecodecs with pip")
        package = "imagecodecs"
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        import imagecodecs


def get_py_slices(
    subb, requested_start, requested_size_on_level0, output_downscale_factor: int = 1
):
    """


    return: isconj, slice_subb, slice_requested, real_size_requested, slice_subb_not_resized

    slice_sub_not_resized: low resolution indices

    isconj is bool. It is true if the tile is (at least partially) in requested area
    real_size is unpredictable because if output downcale is 2 the division can be odd.


    n - means not resized

    """
    s_st = np.asarray(subb.start[-3:-1])
    s_sh = np.asarray(subb.shape[-3:-1])
    r_st = np.asarray(requested_start)
    r_sh = np.asarray(requested_size_on_level0) * output_downscale_factor
    # r_sh_out = np.asarray(requested_size)
    odf = output_downscale_factor

    # real_start_subb = start_subb - (start_requ + shape_requ)
    # real_end_subb   = start_subb + (shape_subb - start_requ)
    subb_start_before_requ_stop = (r_st + r_sh - s_st) > 0
    requ_start_before_subb_stop = (s_st + s_sh - r_st) > 0
    isconj = (subb_start_before_requ_stop.all()) and requ_start_before_subb_stop.all()

    # TODO the size of the endpoint should be probably +1
    if isconj:
        st_in_s = np.max(np.vstack([r_st - s_st, [0, 0]]), axis=0).astype(int)
        sp_in_s = np.min(np.vstack([r_st + r_sh - s_st, s_sh]), axis=0).astype(int)
        st_in_sn = np.max(np.vstack([(r_st - s_st) / odf, [0, 0]]), axis=0).astype(int)
        sp_in_sn = np.min(
            np.vstack([(r_st + r_sh - s_st) / odf, s_sh / odf]), axis=0
        ).astype(int)
        st_in_r = np.max(np.vstack([(s_st - r_st) / odf, [0, 0]]), axis=0).astype(int)
        sp_in_r = np.min(
            np.vstack([(s_st - r_st + s_sh) / odf, r_sh / odf]), axis=0
        ).astype(int)
        # if ((s_st - r_st) % odf != [0, 0]).any():
        #     logger.warning("Problem with downlscale factor. Indices should be int")
        # if (r_sh % odf != [0, 0]).any():
        #     logger.warning("Problem with downlscale factor. Indices should be int")
        sl_s = (slice(st_in_s[0], sp_in_s[0]), slice(st_in_s[1], sp_in_s[1]))
        sl_r = (slice(st_in_r[0], sp_in_r[0]), slice(st_in_r[1], sp_in_r[1]))
        size_r = (-(st_in_r[0] - sp_in_r[0]), -(st_in_r[1] - sp_in_r[1]))
        sl_sn = (slice(st_in_sn[0], sp_in_sn[0]), slice(st_in_sn[1], sp_in_sn[1]))
        size_rn = (-(st_in_sn[0] - sp_in_sn[0]), -(st_in_sn[1] - sp_in_sn[1]))
        sl_rn = (
            slice(st_in_r[0], st_in_r[0] + size_rn[0]),
            slice(st_in_r[1], st_in_r[1] + size_rn[1]),
        )
        # output slice subb no resize

    else:
        sl_s = None
        sl_r = None
        size_r = None
        sl_sn = None
        size_rn = None
        sl_rn = None
    return isconj, sl_s, sl_r, size_r, sl_sn, size_rn, sl_rn


def read_region_with_level(
    czi,
    location,
    size,
    level=0,
    report=None,
    # use_resize_from_level0=False
):
    """
    Read region from czi file. White color is filled where no pixels are given if the
    datatype is uint8 or float.

    :param czi: czifile.CziFile(filename) object
    :param location:  two numbers. It can be negative
    :param size: size on output resolution given by downscale factor
    :param downscale_factor: it is di
    :return:
    """
    downscale_factor = int(2**level)
    requested_start = location
    requested_size = size
    value = 0
    if czi.dtype == np.uint8:
        value = 255
    elif czi.dtype == np.float:
        value = 1

    t_res = 0.0
    t_slices = 0.0
    t0 = time.time()
    output = np.full(
        list(requested_size) + [czi.shape[-1]], fill_value=value, dtype=czi.dtype
    )
    number_of_valid_blocks = 0
    number_of_overlapping_blocks = 0
    # subbs = []
    for subb in czi.subblocks():
        t00 = time.time()
        isconj, sl_s, sl_r, sz_r, sl_sn, sz_sn, sl_rn = get_py_slices(
            subb,
            requested_start,
            requested_size,
            output_downscale_factor=downscale_factor,
        )
        t01 = time.time()
        t_slices += float(t01 - t00)

        if isconj:
            logger.trace("Whole czi block is inside requested region")
            number_of_overlapping_blocks += 1

            # subbs.append(subb)

            #             plt.figure()
            #             plt.imshow(subb.data()[0,0,0,:,:,:])
            #             plt.title(f"{subb.start}, {subb.shape}, {subb.stored_shape}")
            # there are several blocks covering the location. Their resolution is the same but the size differes.

            #             print(f"{subb.start}, {subb.shape}, {subb.stored_shape}, [{sl_s[0].start}:{sl_s[0].stop}, {sl_s[1].start}:{sl_s[1].stop}], [{sl_r[0].start}:{sl_r[0].stop}, {sl_r[1].start}:{sl_r[1].stop}]")

            # this generates the data with maximal possible resolution. Slow.
            # if subb.shape == subb.stored_shape:
            # this is fast
            subb_shape = subb.shape[-3:-1]
            stored_shape = tuple(np.asarray(subb.stored_shape) * 2**level)[-3:-1]
            if subb_shape == stored_shape:  # subb.stored_shape:
                sd = subb.data(resize=False)
                img = sd[..., sl_sn[0], sl_sn[1], :]
                # sd = subb.data()
                # img_old = sd[..., sl_s[0], sl_s[1], :]
                # logger.debug(img.shape)
                axlist = tuple(range(img.ndim - 3))
                # logger.debug(axlist)
                img = np.squeeze(img, axis=axlist)
                output[sl_rn] = img
                number_of_valid_blocks += 1

            # elif use_resize_from_level0:
            #     sd = subb.data(resize=False)
            #     img = sd[..., sl_sn[0], sl_sn[1], :]
            #     # try:
            #     if True:
            #         if np.asarray(sz_r != 0).all():
            #             osh = (sz_r[0], sz_r[1], sd.shape[-1])
            #             # t00 = time.time()
            #             axlist = tuple(range(img.ndim - 3))
            #             # logger.debug(axlist)
            #             img = np.squeeze(img, axis=axlist)
            #             img_smaller = skimage.transform.resize(
            #                 img,
            #                 output_shape=osh,
            #                 preserve_range=True
            #             ).astype(img.dtype)
            #             output[sl_r] = img_smaller
            # t01 = time.time()
            # t_res += float(t01 - t00)
            #         logger.trace(f"osh={osh}, im.sh={img.shape}") #[{sl_s[0].start}:{sl_s[0].stop}, {sl_s[1].start}:{sl_s[1].stop}], [{sl_r[0].start}:{sl_r[0].stop}, {sl_r[1].start}:{sl_r[1].stop}]")
            #     logger.trace(f"{subb.start}, {subb.shape}, {subb.stored_shape}") #[{sl_s[0].start}:{sl_s[0].stop}, {sl_s[1].start}:{sl_s[1].stop}], [{sl_r[0].start}:{sl_r[0].stop}, {sl_r[1].start}:{sl_r[1].stop}]")
            # except OverflowError as e:
            #     import traceback
            #     from PyQt5.QtCore import pyqtRemoveInputHook
            #     pyqtRemoveInputHook()
            #     logger.debug(traceback.format_exc())
            #     import pdb
            #     pdb.set_trace()
            # threr are almost same outputs. The difference is in size of the images
            # there can be 1 pixel error due to integer division
            # img_smaller_alternative = skimage.transform.downscale_local_mean(
            #     img,
            #     factors=(downscale_factor, downscale_factor, 1))
            # print(
            #     f"{subb.start}, {subb.shape}, {subb.stored_shape}, [{sl_s[0].start}:{sl_s[0].stop}, {sl_s[1].start}:{sl_s[1].stop}], [{sl_r[0].start}:{sl_r[0].stop}, {sl_r[1].start}:{sl_r[1].stop}]")
    #             break
    #         else:
    #             logger.debug(f" not equal size of subb {subb.start}, {subb.shape}")
    t1 = time.time()
    logger.trace(
        f"time to get region={t1-t0}, cumulative resize time={t_res}, cumulative get slices time={t_slices}"
    )
    if number_of_valid_blocks == 0 and number_of_overlapping_blocks > 0:
        logger.debug(
            f"number of used blocks for raster recontruction = {number_of_valid_blocks}"
        )
        logger.debug(
            f"number of overlapping blocks for recontruction = {number_of_overlapping_blocks}"
        )
        logger.debug(
            f"Failed loading raster on level={level}. Reconstructing raster image by resizing from level=0"
        )
        size0 = np.asarray(size) * downscale_factor

        output0 = read_region_with_level(czi, location, size0, level=0, report=report)
        output = skimage.transform.resize(
            output0, output_shape=output.shape, preserve_range=True
        ).astype(output.dtype)
    # TODO here is potential to extract the data more memory efficient by resizing each block

    return output
