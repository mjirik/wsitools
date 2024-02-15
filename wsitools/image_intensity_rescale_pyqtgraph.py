# /usr/bin/env python
# -*- coding: utf-8 -*-

from . import image_intensity_rescale
from pyqtgraph.parametertree import Parameter
from . import image
from loguru import logger


class RescaleIntensityPercentilePQG:
    def __init__(
        self,
        pname="Intensity Normalization",
        ptype="bool",
        pvalue=False,
        ptip="A preprocessing of input image. It emphasize important structures.",
        pexpanded=False,
    ):
        # self.rescale_intensity_percentile = image_intensity_rescale.RescaleIntensityPercentile()
        params = [
            # {
            #         "name": "Run Intensity Normalization",
            #         "type": "bool",
            #         "tip": "Do the histogram normalization",
            #         "value": False
            #     },
            {
                "name": "Input Low Percentile",
                "type": "int",
                "tip": "Input point for intensity mapping",
                "value": 5,
            },
            {
                "name": "Input High Percentile",
                "type": "int",
                # "tip": "Slope of sigmoidal limit function",
                "tip": "Input point for intensity mapping",
                "value": 95,
            },
            {
                "name": "Low Percentile Mapping",
                "type": "float",
                # "tip": "",
                "value": -0.9,
            },
            {
                "name": "High Percentile Mapping",
                "type": "float",
                # "tip": "Slope of sigmoidal limit function",
                "value": 0.9,
            },
            {
                "name": "Sigmoidal Slope",
                "type": "float",
                "tip": "Slope of sigmoidal limit function",
                "value": 1.0,
            },
        ]
        self.parameters = Parameter.create(
            name=pname,
            type=ptype,
            value=pvalue,
            tip=ptip,
            children=params,
            expanded=pexpanded,
        )

    def set_anim_params(self, anim: image.AnnotatedImage):
        """
        Set parametetrs of AnnotatedImage to intensity rescale.
        :param anim:
        :return:
        """
        # int_norm_params = self.parameters.param("Processing", "Intensity Normalization")
        int_norm_params = self.parameters
        # run_resc_int = int_norm_params.param("Run Intensity Normalization").value()
        # run_resc_int = self.parameters.param("Processing", "Intensity Normalization").value()
        run_resc_int = self.parameters.value()
        logger.debug(f"run rescale intensity: {run_resc_int}")
        anim.set_intensity_rescale_parameters(
            run_intensity_rescale=run_resc_int,
            percentile_range=(
                int_norm_params.param("Input Low Percentile").value(),
                int_norm_params.param("Input High Percentile").value(),
            ),
            percentile_map_range=(
                int_norm_params.param("Low Percentile Mapping").value(),
                int_norm_params.param("High Percentile Mapping").value(),
            ),
            sig_slope=int_norm_params.param("Sigmoidal Slope").value(),
        )
