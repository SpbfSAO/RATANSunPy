import pytest
import numpy as np
from datetime import timedelta
from pathlib import Path
from ratansunpy.time import TimeRange
from ratansunpy.client import SRSClient, RATANClient
from ratansunpy.utils import get_project_root
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy.io import fits
from ratansunpy.core.ar_handling import ARHandler


class TestARHandler:
    @pytest.fixture(scope="class")
    def sample_radio_data(self):
        processed_path = get_project_root()/"data"/"20170905_121217_sun+0_out_processed.fits"
        raw_path = get_project_root()/"data"/"20170905_121217_sun+0_out.fits"
        processed_hdul = fits.open(processed_path)
        raw_hdul = fits.open(raw_path)
        return raw_hdul, processed_hdul

    @pytest.fixture
    def local_srs_base_url(self):
        base_path = get_project_root()/"data"
        base_url = str(base_path) + '/%Y_SRS/%Y%m%dSRS.txt'
        return base_url

    @pytest.fixture
    def sample_calibrated_data(self):
        """Fixture for sample calibrated FITS data."""
        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=np.random.rand(8, 100)),  # I
            fits.ImageHDU(data=np.random.rand(8, 100)),  # V
            fits.ImageHDU(data=np.linspace(1, 18, 10)),  # FREQ
            fits.ImageHDU(data=np.ones((10, 100), dtype=bool))  # Mask
        ])
        hdul[0].header['CDELT1'] = 0.1
        hdul[0].header['CRPIX1'] = 50
        return hdul



    def test_ar_handler_init(self, sample_radio_data, local_srs_base_url):
        raw_hdul, processed_hdul = sample_radio_data
        ratan_client = RATANClient()
        srs_table = ratan_client.form_srstable_with_time_shift(processed_hdul, local_srs_base_url)

        ar_handler = ARHandler(processed_hdul, srs_table=srs_table)
        assert isinstance(ar_handler.solar_x, np.ndarray)
        ar_handler_with_url = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        assert ar_handler.window_size == 100  # Default window size
        assert ar_handler.srs_table['Latitude'].max() == ar_handler_with_url.srs_table['Latitude'].max()

    def test_extract_ar_data_with_window(self, sample_radio_data, local_srs_base_url):
        """Test interval calculation for active regions."""
        raw_hdul, processed_hdul = sample_radio_data

        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        spectrum_data = handler.extract_ar_data_with_window(latitude=408.69)
        assert spectrum_data.shape == (2, 84, 101)  # 2 * window_size + 1
    def test_check_bad_data(self, sample_radio_data, local_srs_base_url):
        """Test interval calculation for active regions."""
        raw_hdul, processed_hdul = sample_radio_data

        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        spectrum_data = handler.extract_ar_data_with_window(latitude=408.69)
        (p99, p100) = handler.check_bad_data(spectrum_data)
        assert p100 > p99

    def test_compute_ar_mask(self, sample_radio_data, local_srs_base_url):
        raw_hdul, processed_hdul = sample_radio_data

        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        spectrum_data = handler.extract_ar_data_with_window(latitude=408.69)

        mask = handler.compute_ar_mask(spectrum_data)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == "bool"



    def test_vis_ar_2d(self, sample_radio_data, local_srs_base_url):
        raw_hdul, processed_hdul = sample_radio_data

        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        spectrum_data = handler.extract_ar_data_with_window(latitude=408.69)
        fig = handler.vis_ar_2d(spectrum_data=spectrum_data, value='V')

        assert True