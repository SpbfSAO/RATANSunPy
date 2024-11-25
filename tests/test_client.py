import pytest
import numpy as np
from datetime import timedelta
from pathlib import Path
from ratansunpy.time import TimeRange
from ratansunpy.client import SRSClient, RATANClient
from ratansunpy.utils import get_project_root
from urllib.request import urlopen
from astropy.table import  Table
from astropy.io import fits


@pytest.fixture
def tr():
    """
    Fixture that provides two Time objects that are more than 1 nanosecond apart.
    """
    tr = TimeRange('2017-09-03', '2017-09-03')
    return tr

@pytest.fixture
def srs():
    """
    Fixture that provides two Time objects that are more than 1 nanosecond apart.
    """
    srs = SRSClient()
    return srs

@pytest.fixture
def ratan_client():
    return RATANClient()

@pytest.fixture
def r_fits_url():
    ratan_client = RATANClient()
    tr = TimeRange('2017-09-03', '2017-09-03')
    url = ratan_client.acquire_data(tr)[0]
    return url

@pytest.fixture
def raw_fits_data_path():
    return get_project_root() / 'data' / '20170903_121257_sun+0_out.fits'


class TestSRSClient:

    def test_acquire_data(self, tr, srs):
        file_urls = srs.acquire_data(tr)
        assert len(file_urls) == 1
        assert isinstance(file_urls[0], str)

    def test_extract_lines(self, tr, srs):

        file_url = srs.acquire_data(tr)[0]
        with urlopen(file_url) as response:
            content = response.read().decode('utf-8').split('\n')
            header, section_lines, supplementary_lines = srs.extract_lines(content)
        assert len(header) == 11
        assert len(section_lines) == 3

    def test_form_data(self, tr,srs):
        file_url = srs.acquire_data(tr)
        data = srs.form_data(file_url)
        assert len(data) == 4
        assert isinstance(data, Table)

    def test_get_data(self, tr, srs):
        data = srs.get_data(tr)
        assert len(data) == 4
        assert isinstance(data, Table)


class TestRATANClient:

    def test_acquire_data(self, tr, ratan_client):
        urls = ratan_client.acquire_data(tr)
        assert len(urls) == 1
        assert isinstance(urls[0], str)

    def test_acquire_data_3_days(self, tr, ratan_client):
        new_tr = TimeRange(tr.start, tr.start+timedelta(days=3))
        urls = ratan_client.acquire_data(new_tr)
        assert len(urls) == 4

    def test_get_scans(self, tr, ratan_client):
        tables = ratan_client.get_scans(tr)
        assert len(tables) == 1
        assert isinstance(tables, Table)

    def test_process_fits_data(self, ratan_client, r_fits_url):

        raw, processed = ratan_client.process_fits_data(r_fits_url,
                                                   save_path=None,
                                                   save_with_original=False)
        assert isinstance(processed, fits.hdu.hdulist.HDUList)
        assert isinstance(raw, fits.hdu.hdulist.HDUList)
        assert processed[1].header['NAXIS'] == 2
        assert processed[1].name == 'I'
        assert processed[2].header['NAXIS'] == 2
        assert processed[2].name == 'V'

    def test_process_fits_data_with_saving(self, ratan_client, r_fits_url):
        save_path = Path(__file__).absolute().parents[1]/'data'
        raw, processed = ratan_client.process_fits_data(r_fits_url,
                                                       save_path=save_path,
                                                       save_with_original=False,
                                                       save_raw=True
                                                    )
        assert isinstance(processed, fits.hdu.hdulist.HDUList)

    def test_process_fits_data_from_disk(self, ratan_client):
        save_path = Path(__file__).absolute().parents[1]/'data'
        data_path = Path(__file__).absolute().parents[1]/'data'/'20170903_121257_sun+0_out.fits'
        raw, processed = ratan_client.process_fits_data(data_path,
                                                       save_path=save_path,
                                                       save_with_original=False,
                                                       save_raw=False
                                                    )
        assert processed[1].data.shape == (84, 3000)
        assert isinstance(processed, fits.hdu.hdulist.HDUList)

    def test_process_fits_with_period(self, tr, ratan_client):
        new_tr = TimeRange("2017-09-03", "2017-09-04")
        save_path = Path(__file__).absolute().parents[1]/'data'
        _, list_phdul = ratan_client.process_fits_with_period(new_tr,
                                                                  save_path=save_path,
                                                                  save_with_original=False,
                                                                  save_raw=True)
        assert isinstance(list_phdul, list)
        assert isinstance(list_phdul[0], fits.HDUList)

    def test_form_srstable_with_time_shift(self, ratan_client, r_fits_url):
        #first url from '2017-09-03'
        _ , processed = ratan_client.process_fits_data(r_fits_url,
                                                        save_path=None,
                                                        save_with_original=False)
        srs_table = ratan_client.form_srstable_with_time_shift(processed)
        latitude = [59.0928548152582, -327.04700624851887, 808.1400074522206, 672.3535489330781]
        longitude = [-282.6809474507876, -0.6094350999806579, 106.6366431326339, -7.442298067666002]
        assert np.allclose(srs_table['Latitude'], latitude)
        assert np.allclose(srs_table['Longitude'], longitude)
        assert len(srs_table) == 4
        assert isinstance(srs_table, Table)

    def test_active_regions_search(self, srs, ratan_client, raw_fits_data_path):
        raw, processed = ratan_client.process_fits_data(raw_fits_data_path,
                                                        save_path=None,
                                                        save_with_original=False)
        srs_table = ratan_client.form_srstable_with_time_shift(processed)

        CDELT1 = processed[0].header['CDELT1']
        CRPIX = processed[0].header['CRPIX1']
        FREQ = processed[3].data
        bad_freq = np.isin(FREQ, [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062])
        I = processed[1].data
        V = processed[2].data
        mask = processed[4].data.astype(bool)
        I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]
        x = np.linspace(
            - CRPIX * CDELT1,
            (V.shape[1] - CRPIX) * CDELT1,
            num=V.shape[1]
        )

        ar_info = ratan_client.active_regions_search(srs_table, x, V, mask)

        assert isinstance(ar_info, Table)

    def test_get_ar_info_from_processed(self,ratan_client,
                                        raw_fits_data_path):
        hdul = ratan_client.get_ar_info_from_processed(str(raw_fits_data_path))
        assert hdul[1].data[0]['Number'] == 2673
        assert hdul[1].data[0]['TotalFlux'].shape == (84,)
        assert hdul[1].data[0]['MaxAmplitude'].shape == (84,)
        assert hdul[1].data[0]['MaxLat'].shape == (84,)
        assert hdul[1].data[0]['MinLat'].shape == (84,)

        assert isinstance(hdul, fits.HDUList)


    def test_get_local_sources_info_from_processed_path(self,
                                                        ratan_client,
                                                        raw_fits_data_path):
        sources_info = ratan_client.get_local_sources_info_from_processed(str(raw_fits_data_path))
        assert isinstance(sources_info, Table)






    def test_get_data(self, tr, ratan_client):
        ar_data = ratan_client.get_data(tr)

        assert isinstance(ar_data, Table)


