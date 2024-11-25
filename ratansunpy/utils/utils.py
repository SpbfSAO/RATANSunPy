import numpy as np
import pywt
from typing import Union, Any
from scipy import ndimage

from numpy import signedinteger


from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def SunSolidAngle(R: float) -> float:
    """
     Calculate the solid angle subtended by the Sun as seen from Earth.

     :param R: Radius of the Sun's disc in arcseconds.
     :type R: float
     :return: Solid angle in degrees.
     :rtype: float

     :Example:

     >>> SunSolidAngle(960)
     6.805218475785968e-05
     """
    R_deg = (R / 3600)
    return np.pi * (np.pi * R_deg / 180) ** 2


def gauss2d(x: Union[np.ndarray, list], y: Union[np.ndarray, list],
            amplitude_x: float, amplitude_y: float,
            mean_x: float, mean_y: float,
            sigma_x: float, sigma_y: float) -> np.ndarray:
    """
        Generate a 2D Gaussian distribution.

        :param x: Array of x coordinates.
        :type x: numpy.ndarray or list of floats
        :param y: Array of y coordinates.
        :type y: numpy.ndarray or list of floats
        :param amplitude_x: Amplitude along the x-axis.
        :type amplitude_x: float
        :param amplitude_y: Amplitude along the y-axis.
        :type amplitude_y: float
        :param mean_x: Mean of the Gaussian along the x-axis.
        :type mean_x: float
        :param mean_y: Mean of the Gaussian along the y-axis.
        :type mean_y: float
        :param sigma_x: Standard deviation along the x-axis.
        :type sigma_x: float
        :param sigma_y: Standard deviation along the y-axis.
        :type sigma_y: float
        :return: 2D Gaussian array.
        :rtype: numpy.ndarray

        :Example:

        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.linspace(-1, 1, 100)
        >>> gauss2d(x, y, 1, 1, 0, 0, 0.1, 0.1).shape
        (100, 100)
        """
    x, y = np.meshgrid(x, y)
    g = amplitude_x * amplitude_y * np.exp(
        -((x - mean_x) ** 2 / (2 * sigma_x ** 2) + (y - mean_y) ** 2 / (2 * sigma_y ** 2)))
    return g


def gauss1d(x: np.ndarray, amplitude: float,
            mean: float, sigma: float) -> np.ndarray:
    """
        Generate a 1D Gaussian distribution.

        :param x: Array of x coordinates.
        :type x: numpy.ndarray
        :param amplitude: Amplitude of the Gaussian.
        :type amplitude: float
        :param mean: Mean of the Gaussian.
        :type mean: float
        :param sigma: Standard deviation of the Gaussian.
        :type sigma: float
        :return: 1D Gaussian array.
        :rtype: numpy.ndarray

        :Example:

        >>> x = np.linspace(-1, 1, 100)
        >>> gauss1d(x, 1, 0, 0.1).shape
        (100,)
        """
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2))


def gaussian_mixture(params: list[float],
                     x: np.ndarray,
                     y: np.ndarray) -> np.ndarray:
    """
        Model data as a mixture of 1D Gaussians and return the residuals.

        :param params: List of Gaussian parameters [amplitude, mean, sigma, amplitude, mean, sigma, ....].
        :type params: list of floats
        :param x: Array of x coordinates.
        :type x: numpy.ndarray
        :param y: Array of observed data values.
        :type y: numpy.ndarray
        :return: Residuals between the model and the observed data.
        :rtype: numpy.ndarray

        :Example:

        >>> params = [1, 0, 0.1, 0.5, 0.5, 0.1]
        >>> x = np.linspace(-1, 1, 100)
        >>> y = np.ones(100)
        >>> gaussian_mixture(params, x, y).shape
        (100,)
        """
    model = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude = params[i]
        mean = params[i + 1]
        stddev = params[i + 2]
        model += gauss1d(x, amplitude, mean, stddev)
    return y - model


def create_rectangle(size: int,
                     width: float,
                     height: float) -> np.ndarray:
    """
    Create a 2D rectangle of the given width and height within a square array.

    :param size: Size of the square array (size x size).
    :type size: int
    :param width: Width of the rectangle.
    :type width: float
    :param height: Height of the rectangle.
    :type height: float
    :return: 2D array with the rectangle.
    :rtype: numpy.ndarray

    :Example:

    >>> create_rectangle(100, 50, 20).shape
    (100, 100)
    """
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    x, y = np.meshgrid(x, y)
    rectangle = np.zeros((size, size))
    mask = (abs(x) <= width / 2) & (abs(y) <= height / 2)
    rectangle[mask] = 1
    return rectangle


def create_sun_model(size: int, radius: float) -> np.ndarray:
    """
    Create a 2D circular model representing the Sun's disc.

    :param size: Size of the square array (size x size).
    :type size: int
    :param radius: Radius of the Sun's disc in pixels.
    :type radius: int
    :return: 2D array with the circular Sun model.
    :rtype: numpy.ndarray

    :Example:

    >>> create_sun_model(100, 20).shape
    (100, 100)
    """
    y, x = np.ogrid[-size // 2:size // 2, -size // 2:size // 2]
    mask = x ** 2 + y ** 2 <= radius ** 2
    sun_model = np.zeros((size, size))
    sun_model[mask] = 1
    return sun_model


def calculate_area(image: np.ndarray) -> float:
    """
    Calculate the area under a 1D curve or a 2D surface.

    :param image: Array representing the image or curve.
    :type image: numpy.ndarray
    :return: The calculated area.
    :rtype: float

    :Example:

    >>> calculate_area(np.ones(100))
    99.0
    """
    return float(np.trapz(image))


def bwhm_to_sigma(bwhm: float) -> float:
    """
    Convert beam width at half maximum (BWHM) to the standard deviation (sigma).

    :param bwhm: Beam width at half maximum.
    :type bwhm: float
    :return: Corresponding standard deviation (sigma).
    :rtype: float

    :Example:

    >>> bwhm_to_sigma(2.355) # doctest: +ELLIPSIS
    0.7071608181730225
    """
    fwhm = np.sqrt(1 / 2) * bwhm
    return float(fwhm / (2 * np.sqrt(2 * np.log(2))))


def flip_and_concat(values: np.ndarray,
                    flip_values: bool = False) -> np.ndarray:
    """
    Flip and concatenate an array.

    :param values: Array of values to be flipped and concatenated.
    :type values: numpy.ndarray
    :param flip_values: If True, flips the sign of the flipped values.
    :type flip_values: bool
    :return: Concatenated array.
    :rtype: numpy.ndarray

    :Example:

    >>> flip_and_concat(np.array([1, 2, 3]), True)
    array([-3, -2,  1,  2,  3])
    """
    flipped = -values[::-1][:-1] if flip_values else values[::-1][:-1]
    return np.concatenate((flipped, values))


def error(scale_factor: float,
          experimental_data: np.ndarray,
          theoretical_data: np.ndarray) -> float:
    """
    Calculate the squared error between experimental and theoretical data.

    :param scale_factor: Scaling factor applied to experimental data.
    :type scale_factor: float
    :param experimental_data: Array of experimental data values.
    :type experimental_data: numpy.ndarray
    :param theoretical_data: Array of theoretical data values.
    :type theoretical_data: numpy.ndarray
    :return: The squared error.
    :rtype: float

    :Example:

    >>> error(1, np.ones(100), np.zeros(100)) # doctest: +ELLIPSIS
    100.0
    """
    return float(np.sum((scale_factor * experimental_data - theoretical_data) ** 2))


def wavelet_denoise(data: np.ndarray, wavelet, level):
    """
    Denoise a signal using wavelet decomposition.

    :param data: Array of data values to be denoised.
    :type data: numpy.ndarray
    :param wavelet: Wavelet type to be used for decomposition.
    :type wavelet: str
    :param level: Level of decomposition.
    :type level: int
    :return: Denoised data.
    :rtype: numpy.ndarray

    :Example:

    >>> data = np.random.normal(0, 1, 100)
    >>> wavelet_denoise(data, 'sym6', 2).shape
    (100,)
    """
    N = 2 ** np.ceil(np.log2(len(data))).astype(int)
    data_padded = np.pad(data, (0, N - len(data)), 'constant', constant_values=(0, 0))
    coeff = pywt.wavedec(data_padded, wavelet, mode="sym")
    sigma = np.median(np.abs(coeff[-level] - np.median(coeff[-level]))) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(data_padded)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode="sym")[:len(data)]


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = ndimage.zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = ndimage.zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)

        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out
