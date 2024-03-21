import numpy as np


def sigmoidal(x):
    z = 1 / (1 + np.exp(-x))
    return z


def make_norm(imdata, mn, st):
    imdata_norm = (imdata - mn) / st
    return imdata_norm


def hist_mean_std_normalization(imdata):
    pixeldata = imdata.flatten()
    #     plt.hist(pixeldata, bins=40)
    mn = np.mean(pixeldata)
    st = np.std(pixeldata)
    imdata_norm = make_norm(imdata, mn, st)
    return imdata_norm, mn, st


def rescale_intensity_no_limits(img, in_range, out_range=(-0.9, 0.9)):
    x0, x1 = in_range
    y0, y1 = out_range
    k = (y0 - y1) / (x0 - x1)
    q = y0 - k * x0

    img_out = k * img + q
    return img_out


# def rescale_intensity_by_percentile(img, percentile_range=(5, 95), sig_range=(-0.9, 0.9), sig_slope=1):
#     p2, p98 = np.percentile(img, percentile_range)
#     imout = rescale_intensity_no_limits(img, in_range=(p2, p98), out_range=sig_range)
#     return sigmoidal(img * sig_slope)


class RescaleIntensityPercentile:
    def __init__(self):
        self.percentile_range = None
        self.percentile_map_range = None
        self.sig_slope = None
        # self.input_dtype = None
        self.set_parameters()

    def set_parameters(
        self, percentile_range=(5, 95), percentile_map_range=(-0.9, 0.9), sig_slope=1
    ):
        self.percentile_range = percentile_range
        self.percentile_map_range = percentile_map_range
        self.sig_slope = sig_slope

    def calculate_intensity_dependent_parameters(self, img):
        self.percentile_range_values = np.percentile(img, self.percentile_range)
        # self.input_dtype = img.dtype

    def rescale_intensity(self, img):
        input_dtype = img.dtype
        imgout = rescale_intensity_no_limits(
            img,
            in_range=self.percentile_range_values,
            out_range=self.percentile_map_range,
        )
        imgout = sigmoidal(imgout * self.sig_slope)
        if input_dtype == np.uint8:
            imgout = imgout * 255
        return imgout.astype(input_dtype)
