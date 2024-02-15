#! /usr/bin/python
# -*- coding: utf-8 -*-

from loguru import logger
import unittest
import os.path as op
import pytest

path_to_script = op.dirname(op.abspath(__file__))

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
import scaffan
import scaffan.image as scim


@pytest.fixture
def anim_scp003():
    fn = io3d.datasets.join_path(
        "medical", "orig", "sample_data", "SCP003", "SCP003.ndpi", get_root=True
    )
    anim = scim.AnnotatedImage(fn)
    return anim


class ImageAnnotationTest(unittest.TestCase):
    def test_get_pixelsize_on_different_levels(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        logger.debug("filename {}".format(fn))
        imsl = openslide.OpenSlide(fn)

        pixelsize1, pixelunit1 = scim.get_pixelsize(imsl)
        self.assertGreater(pixelsize1[0], 0)
        self.assertGreater(pixelsize1[1], 0)

        pixelsize2, pixelunit2 = scim.get_pixelsize(imsl, level=2)
        self.assertGreater(pixelsize2[0], 0)
        self.assertGreater(pixelsize2[1], 0)

        self.assertGreater(pixelsize2[0], pixelsize1[0])
        self.assertGreater(pixelsize2[1], pixelsize1[1])

    def test_anim(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        offset = anim.get_offset_px()
        self.assertEqual(len(offset), 2, "should be 2D")
        im = anim.get_image_by_center((10000, 10000), as_gray=True)
        self.assertEqual(len(im.shape), 2, "should be 2D")

        annotations = anim.read_annotations()
        self.assertGreater(len(annotations), 1, "there should be 2 annotations")
        # plt.figure()
        # plt.imshow(im)
        # plt.show()
        assert im[0, 0] == pytest.approx(
            0.767964705882353, 0.001
        )  # expected intensity is 0.76

    def test_anim(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)

        loc = [10000, 10000]
        center = anim.get_region_center_by_location(
            location=loc, level=2, size=[100, 100]
        )
        loc_out = anim.get_region_location_by_center(
            center=center, level=2, size=[100, 100]
        )
        assert np.array_equal(loc, loc_out)

        # assert np.abs(im[0, 0] - 0.767964705882353) < 0.001  # expected intensity is 0.76

    def test_anim_intensity_rescale(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        offset = anim.get_offset_px()
        anim.set_intensity_rescale_parameters(
            run_intensity_rescale=True,
            percentile_range=(5, 95),
            percentile_map_range=(-0.9, 0.9),
            sig_slope=1,
        )
        self.assertEqual(len(offset), 2, "should be 2D")
        im = anim.get_image_by_center((10000, 10000), as_gray=True)
        self.assertEqual(len(im.shape), 2, "should be 2D")

        annotations = anim.read_annotations()
        self.assertGreater(len(annotations), 1, "there should be 2 annotations")
        # plt.figure()
        # plt.imshow(im)
        # plt.show()
        assert im[0, 0] == pytest.approx(0.055, 0.1)  # expected intensity is 0.76
        # assert np.abs(im[0, 0] - 0.767964705882353) < 0.001  # expected intensity is 0.76

    def test_file_info(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        msg = anim.get_file_info()
        self.assertEqual(type(msg), str)
        self.assertLess(0, msg.find("mm"))

    def test_anim_region(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        anim.set_region_on_annotations(0, 3)
        mask = anim.get_annotation_region_raster(0)
        image = anim.get_region_image()
        plt.imshow(image)
        plt.contour(mask)
        # plt.show()
        self.assertGreater(np.sum(mask), 20)
        assert image[0, 0, 0] == pytest.approx(97, 10)
        assert image[0, 0, 1] == pytest.approx(77, 10)
        assert image[0, 0, 2] == pytest.approx(114, 10)

        image = anim.get_region_image(as_gray=True, as_unit8=True)
        plt.imshow(image)
        plt.contour(mask)
        # plt.show()
        self.assertGreater(np.sum(mask), 20)
        assert image.ndim == 2
        assert image[0, 0] == pytest.approx(97, 10)
        assert image.dtype is np.dtype(np.uint8)

    def test_anim_region_coords_to_global_and_back(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        anim.set_region_on_annotations(0, 3)
        pts_local = np.asarray([[1000, 1000, 2000], [5000, 6000, 5000]])

        pts_global = anim.coords_region_px_to_global_px(pts_local)
        pts_local_out = anim.coords_global_px_to_view_px(pts_global)

        assert np.array_equal(pts_local, pts_local_out)

        # this works on windows
        # assert image[0, 0, 0] == 97
        # assert image[0, 0, 1] == 77
        # assert image[0, 0, 2] == 114

    def test_region_select_by_title(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        anim.set_region_on_annotations("obj1", 3)
        mask = anim.get_annotation_region_raster("obj1")
        image = anim.get_region_image()
        plt.imshow(image)
        plt.contour(mask)
        # plt.show()
        self.assertGreater(np.sum(mask), 20)
        self.assertTrue(
            np.array_equal(mask.shape[:2], image.shape[:2]),
            "shape of mask should be the same as shape of image",
        )
        assert image[0, 0, 0] == pytest.approx(187, 10)

    # def test_region_select_area_definition_in_mm(self):
    #     fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
    #     anim = scim.AnnotatedImage(fn)
    #     anim.set_region_on_annotations("obj1", 3)
    #     mask = anim.get_annotation_region_raster("obj1")
    #     image = anim.get_region_image()
    #     plt.imshow(image)
    #     plt.contour(mask)
    #     # plt.show()
    #     self.assertGreater(np.sum(mask), 20)
    #     self.assertTrue(np.array_equal(mask.shape[:2], image.shape[:2]), "shape of mask should be the same as shape of image")

    def test_select_view_by_title_and_plot(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        annotation_ids = anim.select_annotations_by_title("obj1")
        view = anim.get_views(annotation_ids, margin=0.5)[0]
        image = view.get_region_image()
        plt.imshow(image)
        view.plot_annotations("obj1")
        # plt.show()
        self.assertGreater(image.shape[0], 100)
        mask = view.get_annotation_raster("obj1")
        self.assertTrue(
            np.array_equal(mask.shape[:2], image.shape[:2]),
            "shape of mask should be the same as shape of image",
        )
        assert image[0, 0, 0] == 202

    def test_select_view_by_title_and_plot_floating_resolution(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        annotation_ids = anim.select_annotations_by_title("obj1")
        view = anim.get_views(annotation_ids, margin=0.5)[0]
        pxsize, pxunit = view.get_pixelsize_on_level()
        image = view.get_region_image()
        plt.subplot(221)
        plt.imshow(image)
        view.plot_annotations("obj1")
        plt.suptitle("{} x {} [{}]".format(pxsize[0], pxsize[1], pxunit))

        self.assertGreater(image.shape[0], 100)
        mask = view.get_annotation_raster("obj1")
        self.assertTrue(
            np.array_equal(mask.shape[:2], image.shape[:2]),
            "shape of mask should be the same as shape of image",
        )
        plt.subplot(222)
        plt.imshow(mask)
        assert np.sum(mask == 1) == 874739  # number of expected pixels in mask

        view2 = view.to_pixelsize(pixelsize_mm=[0.01, 0.01])
        image2 = view2.get_region_image()
        plt.subplot(223)
        plt.imshow(image2)
        view2.plot_annotations("obj1")
        mask = view2.get_annotation_raster("obj1")
        plt.subplot(224)
        plt.imshow(mask)
        self.assertTrue(
            np.array_equal(mask.shape[:2], image2.shape[:2]),
            "shape of mask should be the same as shape of image",
        )
        assert np.sum(mask == 1) == 1812

        # plt.show()

    def test_merge_views(self):
        """
        Create two views with based on same annotation with different margin. Resize the inner to low resolution.
        Insert the inner image with low resolution into big image on high resolution.
        :return:
        """
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        annotation_ids = anim.select_annotations_by_title("obj1")
        view1 = anim.get_views(annotation_ids, margin=1.0, pixelsize_mm=[0.005, 0.005])[
            0
        ]
        image1 = view1.get_region_image()
        # plt.imshow(image1)
        # plt.colorbar()
        # plt.show()

        view2 = anim.get_views(annotation_ids, margin=0.1, pixelsize_mm=[0.05, 0.05])[0]
        image2 = view2.get_region_image()
        # plt.imshow(image2)
        # plt.show()
        logger.debug(
            f"Annotation ID: {annotation_ids}, location view1 {view1.region_location}, view2 {view2.region_location}"
        )

        merged = view1.insert_image_from_view(view2, image1, image2)
        # plt.imshow(merged)
        # plt.show()
        # diffim = image1[:, :, :3].astype(np.int16) - merged[:, :, :3].astype(np.int16)
        diffim = image1[:, :, :3].astype(int) - merged[:, :, :3].astype(int)
        errimg = np.mean(np.abs(diffim), 2)

        err = np.mean(errimg)
        self.assertLess(
            err, 3, "Mean error in intensity levels per pixel should be low"
        )
        self.assertLess(
            1,
            err,
            "Mean error in intensity levels per pixel should be low but there should be some error.",
        )

    def test_view_margin_size(self):
        """
        Compare two same resolution images with different margin
        :return:
        """
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        annotation_ids = anim.select_annotations_by_title("obj1")

        img1 = anim.get_views(annotation_ids, margin=0.0, pixelsize_mm=[0.005, 0.005])[
            0
        ].get_region_image(as_gray=True)
        img2 = anim.get_views(annotation_ids, margin=1.0, pixelsize_mm=[0.005, 0.005])[
            0
        ].get_region_image(as_gray=True)

        sh1 = np.asarray(img1.shape)
        sh2 = np.asarray(img2.shape)
        self.assertTrue(
            np.all((sh1 * 2.9) < sh2),
            "Boundary adds 2*margin*size of image to the image size",
        )
        self.assertTrue(
            np.all(sh2 < (sh1 * 3.1)),
            "Boundary adds 2*margin*size of image to the image size",
        )

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(img1)
        # plt.figure()
        # plt.imshow(img2)
        # plt.show()

        # test that everything is still pixel-precise
        assert img1[0, 0] == pytest.approx(
            0.47553590, 0.001
        )  # expected intensity is 0.76
        assert img2[0, 0] == pytest.approx(0.765724, 0.001)

    def test_select_view_by_center_mm(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        size_on_level = [100, 100]
        view = anim.get_view(
            center_mm=[10, 11],
            size_on_level=size_on_level,
            level=5,
            margin=0,
            # size_mm=[0.1, 0.1]
        )
        image = view.get_region_image()
        logger.debug(f"location: {view.region_location}")
        logger.debug(f"pixelsize: {view.region_pixelsize}")
        plt.imshow(image)
        # view.plot_annotations("obj1")
        # plt.show()

        assert np.array_equal(image.shape[:2], size_on_level)

    def test_select_view_by_loc_mm(self):
        fn = io3d.datasets.join_path("medical", "orig", "CMU-1.ndpi", get_root=True)
        anim = scim.AnnotatedImage(fn)
        size_on_level = [100, 100]
        view = anim.get_view(
            location_mm=[10, 11],
            size_on_level=size_on_level,
            level=5,
            # size_mm=[0.1, 0.1]
        )
        image = view.get_region_image()
        logger.debug(f"location: {view.region_location}")
        logger.debug(f"pixelsize: {view.region_pixelsize}")
        # plt.imshow(image)
        # view.plot_annotations("obj1")
        # plt.show()

        assert np.array_equal(image.shape[:2], size_on_level)
        # self.assertGreater(image.shape[0], 100)
        # mask = view.get_annotation_region_raster("obj1")
        # self.assertTrue(np.array_equal(mask.shape[:2], image.shape[:2]),
        #                 "shape of mask should be the same as shape of image")
        # assert image[0, 0, 0] == 202


def test_outer_and_inner_annotation(anim_scp003):
    anim = anim_scp003
    ann_ids = anim.get_annotations_by_color("#00FFFF")
    assert len(ann_ids) > 0
    assert len(ann_ids) == 3

    # find outer annotation from 0th cyan object
    outer_id = anim.select_outer_annotations(ann_ids[0])
    assert len(outer_id) == 1

    # find inner annotations to outer annotation of 0th object
    inner_ids = anim.select_inner_annotations(outer_id[0])
    assert len(inner_ids) == 2

    # find black inner annotations to outer annotation of 0th object
    inner_ids = anim.select_inner_annotations(outer_id[0], color="#000000")
    assert len(inner_ids) == 1

    cyan_inner_ids = anim.select_inner_annotations(outer_id[0], color="#00FFFF")
    assert len(cyan_inner_ids) == 1
    assert ann_ids[0] == cyan_inner_ids[0]

    black_ann_ids = anim.get_annotations_by_color("#000000")
    # find black inner annotations to outer annotation of 0th object
    inner_ids = anim.select_inner_annotations(outer_id[0], ann_ids=black_ann_ids)
    assert len(inner_ids) == 1


def test_get_outer_ann(anim_scp003):
    anim = anim_scp003
    color = "#000000"
    outer_ids, holes_ids = anim.select_just_outer_annotations(color, return_holes=True)
    logger.debug(f"outer ids {outer_ids}")
    logger.debug(f"holes ids {holes_ids}")
    assert len(outer_ids) > 0
    assert len(holes_ids) > 0
    assert len(holes_ids[0]) > 0
    assert (
        anim.select_inner_annotations(outer_ids[0], color=color)[0] == holes_ids[0][0]
    )


def test_get_annotation_center(anim_scp003):
    anim = anim_scp003
    center_x, center_y = anim.get_annotation_center_mm(1)
    # print(anim.annotations[1]["y_mm"])
    # print(anim.annotations[1]["x_mm"])
    # for some strange reason the y is usually negative
    assert center_x > -100
    assert center_y > -100
    assert center_x < 100
    assert center_y < 100


def test_get_ann_by_color(anim_scp003):
    anim = anim_scp003
    ann_ids_black = anim.get_annotations_by_color("#000000")
    #
    # import scaffan.image
    # from unittest.mock import patch
    # original_foo = scaffan.image.AnnotatedImage.select_annotations_by_color
    # print("staaaart")
    # with patch.object(scaffan.image.AnnotatedImage, 'select_annotations_by_color', autospec=True) as mock_foo:
    #     def side_effect(*args, **kwargs):
    #         logger.debug("mocked function select_annotations_by_color()")
    #         original_list = original_foo(*args, **kwargs)
    #         logger.debug(f"original ann_ids={original_list}")
    #         print(f"original ann_ids={original_list}")
    #         new_list = [original_list[-1]]
    #         logger.debug(f"new ann_ids={new_list}")
    #         return new_list
    #
    #     mock_foo.side_effect = side_effect
    #     ann_ids_black = anim.select_annotations_by_color("#000000")
    assert 10 in ann_ids_black
    assert 11 in ann_ids_black


def test_just_outer_annotations(anim_scp003):
    anim = anim_scp003
    outer_ids, holes_ids = anim.select_just_outer_annotations("#000000")
    assert 10 in outer_ids
    assert [11] in holes_ids

    views = anim.get_views(outer_ids, level=6)
    # ann_rasters = []
    for id1, id2, view_ann in zip(outer_ids, holes_ids, views):
        ann_raster = view_ann.get_annotation_raster(id1, holes_ids=id2)
        # ann_rasters.append(ann_raster)
        assert np.min(ann_raster) == 0
        assert np.max(ann_raster) > 0


def test_get_all_black_annotations_in_view_around_hole(anim_scp003):

    anim = anim_scp003
    # Find inner hole ids
    outer_ids, holes_ids = anim.select_just_outer_annotations("#000000")

    # take first hole to get view
    view = anim.get_view(annotation_id=holes_ids[0][0], level=5)
    ann_raster = view.get_annotation_raster_by_color("#000000")
    # plt.imshow(ann_raster)
    # plt.show()
    assert np.min(ann_raster) == 0, "in the hole there should be 0"
    assert np.max(ann_raster) > 0, "around the hole there should be 1"


def test_view_by_pixelsize_and_size_on_level(anim_scp003):
    anim = anim_scp003
    ann_ids = anim.get_annotations_by_color("#FFFF00")
    size_px = 224
    # anim.get_views(ann_ids)
    view = anim.get_view(
        annotation_id=ann_ids[0],
        size_on_level=[size_px, size_px],
        pixelsize_mm=[0.01, 0.01],
        margin=0,
    )
    img = view.get_region_image(as_gray=True)
    # plt.imshow(img)
    # plt.show()
    assert img.shape[0] == size_px
    assert img.shape[1] == size_px


def test_view_by_size_mm(anim_scp003):
    anim = anim_scp003
    ann_ids = anim.get_annotations_by_color("#FFFF00")
    size_px = 100
    size_mm = [1, 1]
    # anim.get_views(ann_ids)
    view = anim.get_view(
        annotation_id=ann_ids[0], size_mm=size_mm, pixelsize_mm=[0.01, 0.01]
    )
    img = view.get_region_image(as_gray=True)
    # plt.imshow(img)
    # plt.show()
    # assert img.shape[0] == size_px
    # assert img.shape[1] == size_px
    assert pytest.approx(img.shape[:2], [size_px], abs=2.0)
