import os
import shutil
import pytest
from datetime import datetime
from pathlib import Path
from astropy.time import Time
from ratansunpy.time import TimeRange
from ratansunpy.utils.utils import get_project_root
from scrapper import Scrapper


@pytest.fixture
def tr():
    """
    Fixture that provides two Time objects that are more than 1 nanosecond apart.
    """
    tr = TimeRange('2017-09-03', '2017-09-03')
    return tr

    # Fixture to create temporary SRSTables directory for testing
@pytest.fixture(scope="module")
def setup_srs_directory():
    base_path = Path("SRSTables")
    base_path.mkdir(exist_ok=True)

    years = ["2017", "2018", "2019"]
    file_dates = ["20170105", "20170210", "20170903", "20180115", "20190120", "20190225"]

    for year in years:
        year_path = base_path / f"{year}_SRS"
        year_path.mkdir(exist_ok=True)

        # Create test files with different dates
        for file_date in file_dates:
            if file_date.startswith(year):
                (year_path / f"{file_date}SRS.txt").touch()

    yield base_path

class TestScrapper:

 # Provide base path for the test

    def test_ftpfiles(self, tr):
        base_url = 'ftp://ftp.ngdc.noaa.gov/STP/swpc_products/daily_reports/solar_region_summaries/%Y/%m/%Y%m%dSRS.txt'
        scrapper = Scrapper(base_url)
        scrapper.ftpfiles(tr)
        assert True

    def test_srslocalfiles(self, setup_srs_directory, tr):
        base_path = setup_srs_directory.name
        base_url = base_path + '/%Y_SRS/%Y%m%dSRS.txt'

        """Test the srs_localfiles method to ensure it retrieves correct files."""
        scrapper = Scrapper(baseurl=base_url)

        # Define a timerange
        t_range = TimeRange(Time("2017-01-01"), Time("2017-12-31"))

        # Call the srs_localfiles method
        files = scrapper.srs_localfiles(t_range)

        # Expected files within range
        expected_files = [
            "SRSTables/2017_SRS/20170105SRS.txt",
            "SRSTables/2017_SRS/20170210SRS.txt",
            "SRSTables/2017_SRS/20170903SRS.txt"
        ]

        assert sorted(files) == sorted(expected_files), f"Expected {expected_files}, but got {files}"


    def test_srs_form_fileslist(self, tr, setup_srs_directory):
        base_path = setup_srs_directory.name
        base_url = base_path + '/%Y_SRS/%Y%m%dSRS.txt'

        """Test the srs_localfiles method to ensure it retrieves correct files."""
        scrapper = Scrapper(baseurl=base_url)
        t_range = TimeRange(Time("2017-01-01"), Time("2017-12-31"))

        files = scrapper.form_fileslist(tr)

        # Expected files within range
        expected_files = [
            "SRSTables/2017_SRS/20170903SRS.txt"
        ]

        assert sorted(files) == sorted(expected_files), f"Expected {expected_files}, but got {files}"

    def test_real_path(self):
        root_path = get_project_root()
        base_path = root_path.parent/"GAO/SRSTables"
        base_url = str(base_path) + '/%Y_SRS/%Y%m%dSRS.txt'
        scrapper = Scrapper(baseurl=base_url)
        t_range = TimeRange(Time("2017-01-01"), Time("2017-01-03"))

        # Expected files within range
        expected_files = [
            "/Users/irinaknyazeva/Yandex.Disk.localized/Projects/GAO/SRSTables/2017_SRS/20170101SRS.txt",
            "/Users/irinaknyazeva/Yandex.Disk.localized/Projects/GAO/SRSTables/2017_SRS/20170102SRS.txt",
            "/Users/irinaknyazeva/Yandex.Disk.localized/Projects/GAO/SRSTables/2017_SRS/20170103SRS.txt"
        ]

        files = scrapper.form_fileslist(t_range)

        assert sorted(files) == sorted(expected_files), f"Expected {expected_files}, but got {files}"



