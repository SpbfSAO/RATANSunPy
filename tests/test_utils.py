import numpy as np
from pathlib import Path
from ratansunpy.utils import (
    get_project_root,
    SunSolidAngle,
    gauss2d,
    gauss1d,
    gaussian_mixture,
    create_rectangle,
    create_sun_model,
    calculate_area,
    bwhm_to_sigma,
    flip_and_concat,
    error,
    wavelet_denoise,
)

def test_get_project_root():
    project_root = get_project_root()
    assert project_root.parts[-1] == 'RATANSunPy'

def test_sun_solid_angle():
    angle = SunSolidAngle(960)
    assert np.isclose(angle, 6.805e-05)


def test_gauss2d():
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    result = gauss2d(x, y, 1, 1, 0, 0, 0.1, 0.1)
    assert result.shape == (100, 100)
    assert np.isclose(np.max(result), 1.0, atol=1e-01)


def test_gauss1d():
    x = np.linspace(-1, 1, 100)
    result = gauss1d(x, 1, 0, 0.1)
    assert result.shape == (100,)
    assert np.isclose(np.max(result), 1.0, atol=1e-01)


def test_gaussian_mixture():
    params = [1, 0, 0.1, 0.5, 0.5, 0.1]
    x = np.linspace(-1, 1, 100)
    y = np.ones(100)
    result = gaussian_mixture(params, x, y)
    assert result.shape == (100,)


def test_create_rectangle():
    result = create_rectangle(100, 50, 20)
    assert result.shape == (100, 100)
    assert np.sum(result) == 50 * 20  # area of the rectangle


def test_create_sun_model():
    result = create_sun_model(100, 20)
    assert result.shape == (100, 100)
    assert np.isclose(np.sum(result), np.pi * 20 ** 2, atol=1e1)  # area of the circle


def test_calculate_area():
    result = calculate_area(np.ones(100))
    assert np.isclose(result, 99.0)


def test_bwhm_to_sigma():
    bs = bwhm_to_sigma(2.355)
    assert np.isclose(bs, 0.707, atol=1e-3)


def test_flip_and_concat():
    result = flip_and_concat(np.array([1, 2, 3]), True)
    expected = np.array([-3, -2, 1, 2, 3])
    np.testing.assert_array_equal(result, expected)
    result = flip_and_concat(np.array([1, 2, 3]), False)
    expected = np.array([3, 2, 1, 2, 3])
    np.testing.assert_array_equal(result, expected)


def test_error():
    experimental_data = np.ones(100)
    theoretical_data = np.zeros(100)
    assert error(1, experimental_data, theoretical_data) == 100.0


def test_wavelet_denoise():
    data = np.random.normal(0, 1, 100)
    result = wavelet_denoise(data, 'db1', 2)
    assert result.shape == (100,)
    assert np.var(result) < np.var(data)  # denoised data should have lower variance
