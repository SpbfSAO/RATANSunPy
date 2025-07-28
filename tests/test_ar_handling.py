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
            fits.ImageHDU(data=np.linspace(1, 18, 8)),  # FREQ
            fits.ImageHDU(data=np.ones((8, 100), dtype=int))  # Mask
        ])
        hdul[0].header['CDELT1'] = 0.1
        hdul[0].header['CRPIX1'] = 50
        hdul[0].header['DATE-OBS'] = '2023'
        hdul[0].header['TIME-OBS'] = '19-04'
        hdul[0].header['AZIMUTH'] = 0.1
        hdul[0].header['SOLAR_R'] = 50
        hdul[0].header['SOLAR_B'] = 0.1
        hdul[0].header['SOL_DEC'] = 50
        hdul[0].header['ANGLE'] = 50
        return hdul


    @pytest.fixture
    def sample_srs_table(self):
        """Fixture for sample SRS table."""
        return Table({
            'Number': [1158],
            'Latitude': [0.5],
            'Timestamp': ['20170905_121217']
        })

    @pytest.fixture
    def sample_handler(self, sample_calibrated_data, sample_srs_table):

        handler = ARHandler(sample_calibrated_data, srs_table=sample_srs_table)

        return handler



    def test_ar_handler_init(self, sample_radio_data, local_srs_base_url):
        raw_hdul, processed_hdul = sample_radio_data
        ratan_client = RATANClient()
        srs_table = ratan_client.form_srstable_with_time_shift(processed_hdul, local_srs_base_url)

        ar_handler = ARHandler(processed_hdul, srs_table=srs_table)
        assert isinstance(ar_handler.solar_x, np.ndarray)
        ar_handler_with_url = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        assert ar_handler.window_size == 50  # Default window size
        assert ar_handler.srs_table['Latitude'].max() == ar_handler_with_url.srs_table['Latitude'].max()

    def test_extract_ar_data_with_window(self, sample_radio_data, local_srs_base_url):
        """Test interval calculation for active regions."""
        raw_hdul, processed_hdul = sample_radio_data

        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        spectrum_data = handler.extract_ar_data_with_window(latitude=408.69)
        assert spectrum_data.shape == (2, 84, 101)  # 2 * window_size + 1

    def test_identify_and_replace_outliers(self, sample_handler):
        """Test interval calculation for active regions."""

        handler = sample_handler
        spectrum_data = handler.I
        spectrum_data[4, 20] = 50
        spectrum_data = handler.identify_and_replace_outliers(spectrum_data,
                                                              threshold_multiplier=2)
        assert spectrum_data[4, 20] < 50

    def test_compute_ar_mask(self, sample_radio_data, local_srs_base_url):
        raw_hdul, processed_hdul = sample_radio_data

        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        spectrum_data = handler.extract_ar_data_with_window(latitude=408.69)

        mask = handler.compute_ar_mask(spectrum_data)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == "bool"

    def test_compute_ar_stats(self, sample_handler):
        handler = sample_handler
        spectrum_data = handler.I
        spectrum_data[:, 60:80] = spectrum_data[:, 60:80]+5
        mask = handler.compute_ar_mask(spectrum_data)
        stats = handler.compute_ar_stats(spectrum_data, mask=mask, ax_data=None)
        mean_all = np.mean(spectrum_data, axis=1)
        assert isinstance(stats, dict)
        assert isinstance(stats['mean'], np.ndarray)
        assert stats['mean'].max() > mean_all.max()

    def test_process_one_regions(self, sample_radio_data, local_srs_base_url):
        """Test interval calculation for active regions."""
        raw_hdul, processed_hdul = sample_radio_data

        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        ar_hdul, filename = handler.process_one_regions(latitude=408.69, ar_number='2673')

        assert True

    def test_extract_ars_from_scan(self, sample_radio_data, local_srs_base_url):
        raw_hdul, processed_hdul = sample_radio_data
        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        save_path = get_project_root()/"data"
        ar_data = handler.extract_ars_from_scan(save_path=save_path)

        assert isinstance(ar_data, list)


    def test_vis_ar_2d(self, sample_radio_data, local_srs_base_url):
        raw_hdul, processed_hdul = sample_radio_data

        handler = ARHandler(processed_hdul, srs_base_url=local_srs_base_url)
        spectrum_data = handler.extract_ar_data_with_window(latitude=408.69)
        fig = handler.vis_ar_2d(spectrum_data=spectrum_data, value='V')

        assert True