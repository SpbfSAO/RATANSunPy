from abc import ABCMeta, abstractmethod
import pathlib
from typing import List, Tuple, Any, Optional
from pathlib import Path
from urllib.request import urlopen
from collections import OrderedDict
from datetime import datetime
import astropy.io.ascii
from astropy.table import Column, MaskedColumn, vstack, Table, join
from astropy.io import fits
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, cumulative_trapezoid
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.signal import fftconvolve
from scipy.signal import find_peaks
import numpy as np
from functools import lru_cache
import re
from ratansunpy.scrapper import Scrapper
from ratansunpy.time import TimeRange
from ratansunpy.utils import *

__all__ = ['RATANClient', 'SRSClient']


class BaseClient(metaclass=ABCMeta):
    @abstractmethod
    def acquire_data(self):
        pass

    @abstractmethod
    def form_data(self):
        pass

    @abstractmethod
    def get_data(self):
        pass


class SRSClient(BaseClient):
    base_url = 'ftp://ftp.ngdc.noaa.gov/STP/swpc_products/daily_reports/solar_region_summaries/%Y/%m/%Y%m%dSRS.txt'

    def extract_lines(self, content: str) -> object:
        """

        :type content: str with the source data by link
        """
        section, final_section = [], []

        for i, line in enumerate(content):
            if re.match(r'^(I\.|IA\.|II\.)', line):
                section.append(i)
            if re.match(
                    r'^(III|COMMENT|EFFECTIVE 2 OCT 2000|PLAIN|This message is for users of the NOAA/SEC Space|NNN)',
                    line, re.IGNORECASE):
                final_section.append(i)

        if final_section and final_section[0] > section[-1]:
            section.append(final_section[0])
        header = content[:section[0]] + [content[s] for s in section]
        for line in section:
            content[line] = '# ' + content[line]

        table1 = content[section[0]:section[1]]
        table1[1] = re.sub(r'Mag\s*Type', r'Magtype', table1[1], flags=re.IGNORECASE)
        table2 = content[section[1]:section[2]]
        if len(section) > 3:
            table3 = content[section[2]:section[3]]
            extra_lines = content[section[3]:]
        else:
            table3 = content[section[2]:]
            extra_lines = None
        data = [table1, table2, table3]
        for i, table in enumerate(data):
            if len(table) > 2 and table[2].strip().title() == 'None':
                del table[2]
        return header, data, extra_lines

    def proccess_lines(self, date, key, lines):
        column_mapping = {
            'Nmbr': 'Number',
            'Location': 'Location',
            'Lo': 'Carrington Longitude',
            'Area': 'Area',
            'Z': 'Z',
            'Ll': 'Longitudinal Extent',
            'Nn': 'Number of Sunspots',
            'Magtype': 'Mag Type',
            'Lat': 'Lat'
        }

        column_types = {
            'Number': np.dtype('i4'),
            'Location': np.dtype('U6'),
            'Carrington Longitude': np.dtype('i8'),
            'Area': np.dtype('i8'),
            'Z': np.dtype('U3'),
            'Longitudinal Extent': np.dtype('i8'),
            'Number of Sunspots': np.dtype('i8'),
            'Mag Type': np.dtype('S4'),
            'Lat': np.dtype('i8'),
        }

        if lines:
            raw_data = astropy.io.ascii.read(lines)
            column_names = list(raw_data.columns)
            raw_data.rename_columns(
                column_names, new_names=[column_mapping[col.title()] for col in column_names]
            )

            if len(raw_data) == 0:
                for c in raw_data.itercols():
                    c.dtype = column_types[c._name]
                raw_data.add_column(Column(data=None, name="ID", dtype=('S2')), index=0)
                raw_data.add_column(Column(data=None, name="Date", dtype=('S10')), index=0)
            else:
                raw_data.add_column(Column(data=[key] * len(raw_data), name="ID"), index=0)
                raw_data.add_column(Column(data=[date] * len(raw_data), name="Date"), index=0)
            return raw_data
        return None

    def parse_longitude(self, value):
        longitude_sign = {'W': 1, 'E': -1}
        if "W" in value or "E" in value:
            return longitude_sign[value[3]] * float(value[4:])

    def parse_latitude(self, value):
        latitude_sign = {'N': 1, 'S': -1}
        if "N" in value or "S" in value:
            return latitude_sign[value[0]] * float(value[1:3])

    def parse_location(self, column):
        latitude = MaskedColumn(name='Latitude')
        longitude = MaskedColumn(name='Longitude')

        for i, loc in enumerate(column):
            if loc:
                lat_value = self.parse_latitude(loc)
                long_val = self.parse_longitude(loc)
                latitude = latitude.insert(i, lat_value)
                longitude = longitude.insert(i, long_val)
            else:
                latitude = latitude.insert(i, None, mask=True)
                longitude = longitude.insert(i, None, mask=True)
        return latitude, longitude

    def parse_lat_col(self, column, latitude_column):
        for i, loc in enumerate(column):
            if loc:
                latitude_column.mask[i] = False
                latitude_column[i] = self.parse_latitude(loc)
        return latitude_column

    def acquire_data(self, timerange: TimeRange) -> list[str]:
        """
        Function for collecting data with predefined timerange

        :param timerange: TimeRange object with time span for ARs searching
        :type timerange: TimeRange
        :returns: list of urls with txt files.
        :rtype: list of string
        """
        scrapper = Scrapper(self.base_url)
        return scrapper.form_fileslist(timerange)

    def form_data(self, file_urls):
        total_table, section_lines, final_section_lines = [], [], []
        for file_url in file_urls:
            tables = []
            with urlopen(file_url) as response:
                content = response.read().decode('utf-8').split('\n')
                header, section_lines, supplementary_lines = self.extract_lines(content)
                issued_lines = [line for line in header if 'issued' in line.lower() and line.startswith(':')][0]
                _, date_text = issued_lines.strip().split(':')[1:]
                issued_date = datetime.strptime(date_text.strip(), "%Y %b %d %H%M UTC")
                meta_id = OrderedDict()
                for h in header:
                    if h.startswith(("I.", "IA.", "II.")):
                        pos = h.find('.')
                        id = h[:pos]
                        id_text = h[pos + 2:]
                        meta_id[id] = id_text.strip()

                for key, lines in zip(list(meta_id.keys()), section_lines):
                    raw_data = self.proccess_lines(issued_date.strftime("%Y-%m-%d"), key, lines)
                    tables.append(raw_data)
                stacked_table = vstack(tables)

                if 'Location' in stacked_table.columns:
                    col_lat, col_lon = self.parse_location(stacked_table['Location'])
                    del stacked_table['Location']
                    stacked_table.add_column(col_lat)
                    stacked_table.add_column(col_lon)

                if 'Lat' in stacked_table.columns:
                    self.parse_lat_col(stacked_table['Lat'], stacked_table['Latitude'])
                    del stacked_table['Lat']

            total_table.append(stacked_table)
        return Table(vstack(total_table))

    def get_data(self, timerange):
        """
        Function for collecting data with predefined timerange
        :param timerange:
        :type timerange: TimeRange
        :return: data table
        :rtype: Table
        """
        file_urls = self.acquire_data(timerange)
        return self.form_data(file_urls)


class RATANClient(BaseClient):
    """ Client for processing and analyzing data from the RATAN-600 radio telescope.

    RATANClient class is a complex implementation that extends a base client class for handling and processing data
    related to solar observations using RATAN-600. The class provides a wide range of methods to acquire, process,
    and analyze solar data from FITS (Flexible Image Transport System) files obtained from a given URL. Here’s an
    overview of what each method does:

    Key Components:
    1. Initialization Parameters and Configuration:

        - `base_url`: URL template to fetch solar data.
        - regex_pattern: A regex pattern to match relevant data files.
        - convolution_template and quiet_sun_model: Loaded from Excel files; these are templates or models of solar data
        that will be used in the analysis.

    2.Data Acquisition and Filtering:
        - `acquire_data`: Uses a Scrapper class to form a list of data files for a given
        time range, based on URL and regex pattern.
        - `get_scans`: Fetches data from acquired file URLs, processes the FITS
        files to extract relevant information, and returns a table with structured data.

    3. Data Processing and Analysis:

        3.1 Convolution and Efficiency Calculations:

            - convolve_sun: Simulates the convolution of a solar model with Gaussian and rectangular functions, then
              computes and returns the ratio of areas under these convolved functions.
            - antenna_efficiency: Calculates the antenna efficiency across various frequencies using the convolution data.

        3.2 Calibration:

            - calibrate_QSModel: Calibrates the quiet sun model data against observed scan data, adjusting for flux efficiency
              and other parameters

        3.3 Positional and Rotational Transformations:
            - heliocentric_transform: Converts solar latitude and longitude into heliocentric coordinates.
            - pozitional_rotation: Rotates positional data based on a given angle.
            - differential_rotation: Calculates the differential rotation rate of the sun based on latitude.
            - pozitional_angle: Computes the positional angle of the sun based on azimuth, solar declination, and solar P-angle.

        3.4 Active Regions and Peak Analysis:
            - active_regions_search: Identifies active regions on the sun by comparing observed data to known solar
              active region locations using wavelet denoising and peak detection.
            - make_multigauss_fit: Fits multiple Gaussian functions to data points, extracting amplitude, mean, and
              standard deviation for each detected peak.
            - gauss_analysis: Performs Gaussian analysis on identified active regions and associates various metrics
              (amplitude, mean, flux, etc.) with these regions.




    Attributes
    ----------
    base_url : str
        Base URL template for accessing data files.
    regex_pattern : str
        Regular expression pattern to match file names.
    convolution_template : pd.DataFrame
        Template for convolving the quiet Sun model, loaded from the Excel file.
    quiet_sun_model : pd.DataFrame
        Model for the quiet Sun brightness temperature, loaded from the Excel file.
    """

    base_url = 'http://spbf.sao.ru/data/ratan/%Y/%m/%Y%m%d_%H%M%S_sun+0_out.fits'
    regex_pattern = '((\d{6,8})[^0-9].*[^0-9]0_out.fits)'

    convolution_template = pd.read_excel(Path(__file__).absolute().parent.joinpath('quiet_sun_template.xlsx'))
    quiet_sun_model = pd.read_excel(Path(__file__).absolute().parent.joinpath('quiet_sun_model.xlsx'))

    def process_fits_data(self,
                          path_to_file: str,
                          bad_freq: Optional[list[float]] = None,
                          save_path: Optional[str] = None,
                          save_with_original: bool = False,
                          save_raw: bool = False
                          ) -> Tuple[fits.HDUList, fits.HDUList]:
        """
        Process FITS data from a given URL or file_path from the disk, optionally saving the original and processed data to a file,
        and return the processed data as a FITS HDUList object.

        :param path_to_file: The URL of the FITS file to process or path to file at disk.
        :type path_to_file: str
        :param bad_freq: A list of bad frequencies to exclude from the data acquisition.
        :type bad_freq: list of float
        :param save_path: Path to save the processed FITS data. If None, the processed FITS object is returned.
        :type save_path: Optional[str]
        :param save_with_original: Whether to save the original data or not.
        :type save_with_original: bool
        :param save_raw: Whether to save raw FITS data or not.
         :type save_raw: bool
        :returns: The raw and processed FITS data object.
        :rtype: Tuple[fits.HDUList, fits.HDUList]

        :Example:

        .. code-block:: python


            >>> ratan_client = RATANClient()
            >>> raw, processed = ratan_client.process_fits_data('http://spbf.sao.ru/data/ratan/2017/09/20170903_121257_sun+0_out.fits')
            >>> isinstance(processed, fits.HDUList)
            True
        """
        if bad_freq is None:
            bad_freq = [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062]
        file_name = str(path_to_file).split('/')[-1]
        file_name_processed = file_name.split('.fits')[0] + '_processed.fits'
        hdul = fits.open(path_to_file)
        hdul.verify('fix')
        data = hdul[0].data

        CDELT1 = hdul[0].header['CDELT1']
        CRPIX = hdul[0].header['CRPIX1']
        SOLAR_R = hdul[0].header['SOLAR_R']
        SOLAR_B = hdul[0].header['SOLAR_B']
        FREQ = hdul[1].data['FREQ']
        OBS_DATE = hdul[0].header['DATE-OBS']
        OBS_TIME = hdul[0].header['TIME-OBS']
        bad_freq = np.isin(FREQ, bad_freq)

        AZIMUTH = hdul[0].header['AZIMUTH']
        SOL_DEC = hdul[0].header['SOL_DEC']
        SOLAR_P = hdul[0].header['SOLAR_P']

        angle = self.pozitional_angle(AZIMUTH, SOL_DEC, SOLAR_P)
        N_shape = data.shape[2]
        x = np.linspace(
            - CRPIX * CDELT1,
            (N_shape - CRPIX) * CDELT1,
            num=N_shape
        )
        flux_eficiency = self.antenna_efficiency(FREQ, SOLAR_R)
        mask, I, V = self.calibrate_QSModel(x, data, SOLAR_R, FREQ, flux_eficiency)
        I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]

        # Pack the processed data into FITS format
        if save_with_original:
            primary_hdu = fits.PrimaryHDU(data)
        else:
            primary_hdu = fits.PrimaryHDU()
        I_hdu = fits.ImageHDU(data=I.astype('float32'), name='I')
        V_hdu = fits.ImageHDU(data=V.astype('float32'), name='V')
        freq_hdu = fits.ImageHDU(data=FREQ.astype('float32'), name='FREQ')
        mask_hdu = fits.ImageHDU(data=mask.astype('int8'), name='mask')

        header = primary_hdu.header
        header['CDELT1'] = CDELT1
        header['CRPIX1'] = CRPIX
        header['SOLAR_R'] = SOLAR_R
        header['SOLAR_B'] = SOLAR_B
        header['DATE-OBS'] = OBS_DATE
        header['TIME-OBS'] = OBS_TIME
        header['AZIMUTH'] = AZIMUTH
        header['SOL_DEC'] = SOL_DEC
        header['SOLAR_P'] = SOLAR_P
        header['ANGLE'] = angle

        hdulist = fits.HDUList([primary_hdu, I_hdu, V_hdu, freq_hdu, mask_hdu, hdul[1]])
        hdulist.verify('fix')

        if save_path:
            hdulist.writeto(Path(save_path) / file_name_processed, overwrite=True)
        if save_raw:
            hdul.writeto(Path(save_path) / file_name, overwrite=True)

        return hdul, hdulist

    def process_fits_with_period(self, timerange: TimeRange,
                                 **kwargs) -> Tuple[List[fits.HDUList], List[fits.HDUList]]:
        """
        Process fits files with a period by urls taken from http://spbf.sao.ru/ and return list of raw
        fits files and processed fits, all kwargs from process_fits_data are available,
         such as option for saving processed fits and raw fits to directory.

        :param timerange: Period of time for which scans will be downloaded and processed
        :param kwargs: keyword arguments valid for process _fits_data
        :return: Tuple[List[fits.HDUList], List[fits.HDUList]]

         :Example:

        .. code-block:: python
        >>> time_range = TimeRange("2017-09-03", "2017-09-04")
        >>> ratan_client = RATANClient()
        >>> _, list_phdul = ratan_client.process_fits_with_period(time_range)
        >>> isinstance(list_phdul, list)
        True

        """
        url_list = self.acquire_data(timerange)
        raw_hdul = []
        processed_hdul = []
        for url in url_list:
            hdul, hdulist = self.process_fits_data(url, **kwargs)
            raw_hdul.append(hdul)
            processed_hdul.append(hdulist)
        return raw_hdul, processed_hdul

    def get_local_sources_info_from_processed(self,
                                              pr_data: Union[str, fits.hdu.hdulist.HDUList],
                                              bad_freq: Optional[list[float]] = None) -> Table:
        """
        Compute local sources info from a processed FITS HDUList and NOAA active regions coordnate info
        :param pr_data:  string with path to file or direct url to source fits or fits file
        :type pr_data: str or fits.hdu.hdulist.HDUList
        :param bad_freq:  list of bd frequencies excluded from computation
        :type bad_freq: list of float
        :return: astropy table with local sources info
        :rtype: Table
        """
        if isinstance(pr_data, str):
            _, processed = self.process_fits_data(pr_data,
                                                  save_path=None,
                                                  save_with_original=False)
        elif isinstance(pr_data, fits.hdu.hdulist.HDUList):
            processed = pr_data
        else:
            NotImplementedError

        if bad_freq is None:
            bad_freq = [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062]

        srs_table = self.form_srstable_with_time_shift(processed)

        CDELT1 = processed[0].header['CDELT1']
        CRPIX = processed[0].header['CRPIX1']
        AZIMUTH = processed[0].header['AZIMUTH']
        FREQ = processed[3].data
        bad_freq = np.isin(FREQ, bad_freq)
        I = processed[1].data
        V = processed[2].data
        mask = processed[4].data.astype(bool)
        I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]
        x = np.linspace(
            - CRPIX * CDELT1,
            (V.shape[1] - CRPIX) * CDELT1,
            num=V.shape[1]
        )

        source_info = self.active_regions_search(srs_table, x, V, mask)

        ar_amount = len(source_info)
        source_info.add_column(Column(name='Azimuth', data=[AZIMUTH] * ar_amount, dtype=('i2')), index=1)
        source_info.add_column(
            Column(name='Amplitude', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
        source_info.add_column(
            Column(name='Mean', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
        source_info.add_column(
            Column(name='Sigma', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
        source_info.add_column(
            Column(name='FWHM', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
        source_info.add_column(
            Column(name='Flux', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
        source_info.add_column(
            Column(name='Total Flux', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)],
                   dtype=object))
        source_info.add_column(
            Column(name='Range', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))

        ratan_data = list(zip(FREQ, I, V))
        source_info = self.gauss_analysis(x, ratan_data, source_info)

        return source_info

    def get_ar_info_from_processed(self, pr_data: Union[str, fits.hdu.hdulist.HDUList],
                                   bad_freq: Optional[list[float]] = None) -> fits.HDUList:
        """
        Compute ar parameters (total flux, maximum amplitude, width (in arcsec))  info from
        a processed FITS HDUList and NOAA active regions coordnate info
        :param pr_data:  string with path to file or direct url to source fits or fits file
        :type pr_data: str or fits.hdu.hdulist.HDUList
        :param bad_freq:  list of bd frequencies excluded from computation
        :type bad_freq: list of float
        :return: astropy table with local sources info
        :rtype: Table

        >>> raw_fits_data_path = get_project_root() / 'data' / '20170903_121257_sun+0_out.fits'
        >>> ratan_client = RATANClient()
        >>> hdul = ratan_client.get_ar_info_from_processed(str(raw_fits_data_path))
        >>> assert hdul[1].data[0]['Number'] == 2673
        >>> assert hdul[1].data[0]['TotalFlux'].shape == (84,)
        >>> assert hdul[1].data[0]['MaxAmplitude'].shape == (84,)
        >>> assert hdul[1].data[0]['MaxLat'].shape == (84,)
        >>> assert hdul[1].data[0]['MinLat'].shape == (84,)
        """

        assert isinstance(pr_data, (str, pathlib.PosixPath, fits.hdu.hdulist.HDUList)), \
            'Processed data should be link to file, path to fits data of fits data'
        if isinstance(pr_data, (str, pathlib.PosixPath)):
            _, processed = self.process_fits_data(pr_data,
                                                  save_path=None,
                                                  save_with_original=False)
        else:
            processed = pr_data

        if bad_freq is None:
            bad_freq = [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062]

        srs_table = self.form_srstable_with_time_shift(processed)

        CDELT1 = processed[0].header['CDELT1']
        CRPIX = processed[0].header['CRPIX1']
        AZIMUTH = processed[0].header['AZIMUTH']
        FREQ = processed[3].data
        bad_freq = np.isin(FREQ, bad_freq)
        I = processed[1].data
        V = processed[2].data
        mask = processed[4].data.astype(bool)
        I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]
        x = np.linspace(
            - CRPIX * CDELT1,
            (V.shape[1] - CRPIX) * CDELT1,
            num=V.shape[1]
        )
        primary_hdu = fits.PrimaryHDU(FREQ)
        primary_hdu.header = processed[0].header
        source_info = self.active_regions_search(srs_table, x, V, mask)

        ar_numbers = np.unique(source_info['Number'])
        total_flux_list = []
        max_amplitude_list = []
        max_x_list = []
        min_x_list = []

        for noaa_ar in ar_numbers:
            region_info = source_info[source_info['Number'] == noaa_ar]
            scan_data = list(zip(FREQ, I, V))
            total_flux = np.zeros_like(FREQ)
            max_amplitude = np.zeros_like(FREQ)
            max_x = np.zeros_like(FREQ)
            min_x = np.zeros_like(FREQ)
            for index, elem in enumerate(scan_data):
                freq, I_data, V_data = elem
                gauss_params, y_min, x_range = self.make_multigauss_fit(x, I_data, region_info)
                gauss_params = gauss_params.reshape(-1, 3)
                gauss_params[:, 0] += y_min
                total_flux[index] = np.sum(np.sqrt(2 * np.pi) * gauss_params[:, 0] * gauss_params[:, 2])
                max_amplitude[index] = np.max(gauss_params[:, 0])
                max_x[index] = np.max(x_range)
                min_x[index] = np.min(x_range)

            total_flux_list.append(total_flux)
            max_amplitude_list.append(max_amplitude)
            max_x_list.append(max_x)
            min_x_list.append(min_x)
        ar_info_part1 = srs_table[['RatanTime', 'Number', 'Mag Type', 'Number of Sunspots', 'Area', 'Z']]
        ar_info_part2 = Table([ar_numbers, total_flux_list, max_amplitude_list, max_x_list, min_x_list],
                              names=('Number', 'TotalFlux', 'MaxAmplitude', 'MaxLat', 'MinLat'))
        ar_info = join(ar_info_part1, ar_info_part2, keys='Number')
        return fits.HDUList([primary_hdu, fits.BinTableHDU(ar_info)])
    
    def find_ar_intervals_using_peaks(self, x: np.ndarray, 
                                    y: np.ndarray, 
                                    ar_table: Table) -> Table:
        """
        Identify intervals around active region centers using peak detection

        :param x: Array of x-coordinates, typically representing time or position
        :type x: np.ndarray
        :param y: Array of y-coordinates, representing the signal intensity
        :type y: np.ndarray
        :param ar_table: Table with active region info, including 'Number' and 'Latitude'
        :type ar_table: Table
        :return: Table with active region numbers and calculated intervals
        :rtype: Table
        """
        peaks_indices, _ = find_peaks(y)
        minima_indices, _ = find_peaks(-y)

        ar_intervals = []
        for number, center_x in ar_table[['Number', 'Latitude']]:
            peak_distances = np.abs(x[peaks_indices] - center_x)
            center_index = peaks_indices[np.argmin(peak_distances)]

            left_minima = minima_indices[minima_indices < center_index]
            right_minima = minima_indices[minima_indices > center_index]

            if len(left_minima) > 0:
                left_index = left_minima[-1]
            else:
                left_index = 0 

            if len(right_minima) > 0:
                right_index = right_minima[0]
            else:
                right_index = len(y) - 1 

            ar_interval = (x[left_index], x[right_index])
            ar_intervals.append(ar_interval)

        ar_table_with_intervals = ar_table.copy()
        ar_table_with_intervals.add_column(Column(data=ar_intervals, name='Interval'))
        return ar_table_with_intervals
    
    def compute_fluxes(self, x: np.ndarray, 
                       y: np.ndarray, 
                       ar_table: Table, 
                       mode: str = 'I') -> Table:
        """
        Calculate fluxes for active regions over specified intervals.

        :param x: Array of x-coordinates, typically representing time or position.
        :type x: np.ndarray
        :param y: Array of y-coordinates, representing the signal intensity. Supports 2D arrays.
        :type y: np.ndarray
        :param ar_table: Table containing active region info, including 'Interval'.
        :type ar_table: Table
        :param mode: Mode of flux calculation, defaults to 'I'.
        :type mode: str
        :return: Table with active region info and calculated fluxes.
        :rtype: Table
        """
        y = np.atleast_2d(y)
        cumulative_integral = cumulative_trapezoid(y, x, axis=1, initial=0)

        cum_integral_interp = interp1d(
            x, cumulative_integral, kind='linear', axis=1,
            fill_value="extrapolate", bounds_error=False
        )

        intervals = np.array(ar_table['Interval'])
        left_x = intervals[:, 0].astype(np.float64)
        right_x = intervals[:, 1].astype(np.float64)

        left_x, right_x = np.minimum(left_x, right_x), np.maximum(left_x, right_x)

        cum_int_left = cum_integral_interp(left_x)
        cum_int_right = cum_integral_interp(right_x)

        fluxes = (cum_int_right - cum_int_left)

        col_name = 'Flux_' + mode
        flux_column = Column(data=fluxes.T, name=col_name)
        ar_table_with_fluxes = ar_table.copy()
        ar_table_with_fluxes.add_column(flux_column)
        return ar_table_with_fluxes
    
    def get_ar_info(self, 
                    pr_data: Union[str, fits.hdu.hdulist.HDUList],
                    bad_freq: Optional[list[float]] = None, 
                    **kwargs) -> Table:
        """
        Retrieve Extract active region information from processed FITS data.
        
        :param pr_data: Path to FITS file, URL, or FITS HDUList with processed data.
        :type pr_data: Union[str, fits.hdu.hdulist.HDUList]
        :param kwargs: Additional keyword arguments for processing options.
        :return: FITS HDUList containing frequency data and active region information.
        :rtype: Table
        """
        assert isinstance(pr_data, (str, pathlib.PosixPath, fits.hdu.hdulist.HDUList)), \
            'Processed data should be link to file, path to fits data of fits data'
        if isinstance(pr_data, (str, pathlib.PosixPath)):
            _, processed = self.process_fits_data(pr_data,
                                                  save_path=None,
                                                  save_with_original=False)
        else:
            processed = pr_data

        if bad_freq is None:
            bad_freq = [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062]

        srs_table = self.form_srstable_with_time_shift(processed)

        CDELT1 = processed[0].header['CDELT1']
        CRPIX = processed[0].header['CRPIX1']
        FREQ = processed[3].data
        bad_freq = np.isin(FREQ, bad_freq)
        I = processed[1].data
        V = processed[2].data
        mask = processed[4].data.astype(bool)
        I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]
        x = np.linspace(
            - CRPIX * CDELT1,
            (V.shape[1] - CRPIX) * CDELT1,
            num=V.shape[1]
        )

        primary_hdu = fits.PrimaryHDU(FREQ)
        primary_hdu.header = processed[0].header
        
        ar_intervals = self.find_ar_intervals_using_peaks(x[mask], I[0][mask], srs_table)
        ar_info = self.compute_fluxes(x, I, ar_intervals, mode='I')
        ar_info = self.compute_fluxes(x, V, ar_info, mode='V')
        return fits.HDUList([primary_hdu, fits.BinTableHDU(ar_info)])

    def condition(self, year, month, data_match):
        """
        Formats the data string based on the year and month. Used in Scrapper class
        :param year: The year of the data.
        :type year: str
        :param month:
        :type month: str
        :param data_match: data text from http content
        :type data_match: str
        :return: string with selected data
        """
        if int(year) < 2010 or (int(year) == 2010 and int(month) < 5):
            return f'{year[:2]}{data_match[:-4]}-{data_match[-4:-2]}-{data_match[-2:]}'
        else:
            return f'{data_match[:-4]}-{data_match[-4:-2]}-{data_match[-2:]}'

    def filter(self, url_list):
        pass

    @lru_cache(maxsize=None)
    def convolve_sun(self, sigma_horiz: float, sigma_vert: float, R: float):
        """    Performs convolution of the Sun model with Gaussian and rectangular functions.

        :param sigma_horiz:
        :type sigma_horiz: float
        :param sigma_vert:
        :type sigma_vert: float
        :param R:
        :type R: float
        :return:
        """
        size = 1000
        x = np.linspace(-size // 2, size // 2, size)
        y = np.linspace(-size // 2, size // 2, size)
        gaussian = gauss2d(x, y, 1, 1, 0, 0, sigma_horiz, sigma_vert)
        rectangle = create_rectangle(size, 6 * sigma_horiz, 4 * R)
        sun_model = create_sun_model(size, R)
        # Perform convolutions
        convolved_gaussian = fftconvolve(sun_model, gaussian, mode='same', axes=1) / np.sum(gaussian)
        convolved_rectangle = fftconvolve(sun_model, rectangle, mode='same', axes=1) / np.sum(rectangle)

        convolved_gaussian = convolved_gaussian / np.max(convolved_gaussian)
        convolved_rectangle = convolved_rectangle / np.max(convolved_rectangle)

        conv_g = np.sum(convolved_gaussian, axis=0)
        conv_r = np.sum(convolved_rectangle, axis=0)
        # Calculate areas under the curve
        area_gaussian = calculate_area(conv_g)
        area_rectangle = calculate_area(conv_r)
        # Division of areas
        area_ratio = area_gaussian / area_rectangle
        return area_ratio

    def antenna_efficiency(self, freq: List[float], R: float) -> List[float]:
        """
        Calculates the antenna efficiency for each frequency.

        :param freq:
        :param R:
        :return:
        """
        areas = []
        for f in freq:
            lambda_value = (3 * 10 ** 8) / (f * 10 ** 9) * 1000
            FWHM_h = 0.85 * lambda_value
            FWHM_v = 0.75 * lambda_value * 60

            sigma_h = round(bwhm_to_sigma(FWHM_h), 9)
            sigma_v = round(bwhm_to_sigma(FWHM_v), 9)

            area_ratio = self.convolve_sun(sigma_h, sigma_v, R)
            areas.append(area_ratio)
        return areas

    def calibrate_QSModel(self, x: np.ndarray, scan_data: np.ndarray,
                          solar_r: float, frequency: np.ndarray,
                          flux_eficiency: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calibrates raw scan data with the quiet Sun model.
        :param x: Array of x coordinates
        :type: np.ndarray
        :param scan_data: Scan data array.
        :type: np.ndarray
        :param solar_r:  Solar radius in arcseconds
        :type: float
        :param frequency: Array of frequencies.
        :type: np.ndarray
        :param flux_eficiency: Array of flux efficiencies for each frequency.
        :type: np.ndarray
        :return: calibrated scan: mask, Stocks component I, Stocks component V
        """
        K_b = 1.38 * 10 ** (-23)  # Boltzmann constant
        c = 3 * 10 ** 8  # Speed of light in vacuum
        freq_size = len(frequency)
        model_freq = self.convolution_template.columns[1:].values.astype(float)

        template_freq = self.quiet_sun_model['freq']
        template_val = self.quiet_sun_model['T_brightness']
        real_brightness = interp1d(template_freq, template_val, bounds_error=False, fill_value="extrapolate")
        full_flux = np.column_stack([
            frequency,
            2 * 10 ** 22 * K_b * real_brightness(frequency) * SunSolidAngle(solar_r) * (c / (frequency * 10 ** 9)) ** (
                -2)
        ])

        R = scan_data[:, 0, :] + scan_data[:, 1, :]
        L = scan_data[:, 0, :] - scan_data[:, 1, :]

        columns = self.convolution_template.columns.values[1:].astype(float)
        mask = (x >= -1.0 * solar_r) & (x <= 1.0 * solar_r)

        calibrated_R = np.zeros((freq_size, R.shape[1]), dtype=float)
        calibrated_L = np.zeros((freq_size, L.shape[1]), dtype=float)
        theoretical_new = np.zeros((freq_size, L.shape[1]), dtype=float)

        for freq_num in range(freq_size):
            real_R, real_L = R[freq_num], L[freq_num]
            freq_diff = np.abs(columns - frequency[freq_num])
            freq_template = np.argmin(freq_diff) + 1

            template_values = self.convolution_template.iloc[:, freq_template].values.copy()
            x_values = self.convolution_template.iloc[:, 0].values.copy()

            convolution_template_arcsec = flip_and_concat(template_values)
            x_arcsec = flip_and_concat(x_values, flip_values=True)

            coeff = full_flux[freq_num, 1] * flux_eficiency[freq_num] / trapezoid(convolution_template_arcsec, x_arcsec)
            theoretical_data = np.interp(x, x_arcsec, coeff * convolution_template_arcsec)
            theoretical_new[freq_num] = theoretical_data

            res_R = minimize(error, np.array([1]), args=(real_R[mask], theoretical_data[mask]))
            res_L = minimize(error, np.array([1]), args=(real_L[mask], theoretical_data[mask]))

            calibrated_R[freq_num, :] = real_R * res_R.x
            calibrated_L[freq_num, :] = real_L * res_L.x
        return mask, (calibrated_R + calibrated_L) / 2, (calibrated_R - calibrated_L) / 2

    def heliocentric_transform(self, Lat: np.ndarray, Long: np.ndarray,
                               SOLAR_R: float, SOLAR_B: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms solar coordinates to a heliocentric system.
        :param Lat:  Latitude array in degrees
        :type: np.ndarray
        :param Long: Longitude array in degrees
        :type: np.ndarray
        :param SOLAR_R:  Solar radius.
        :type: float
        :param SOLAR_B: Solar B angle
        :type: float
        :return: Heliocentric x and y coordinates.
        :rtype: Tuple[np.ndarray, np.ndarray]

        """
        return (
            SOLAR_R * np.cos(Lat * np.pi / 180) * np.sin(Long * np.pi / 180),
            SOLAR_R * (np.sin(Lat * np.pi / 180) * np.cos(SOLAR_B * np.pi / 180)
                       - np.cos(Lat * np.pi / 180) * np.cos(Long * np.pi / 180) * np.sin(SOLAR_B * np.pi / 180))
        )

    def pozitional_rotation(self, Lat: np.ndarray, Long: np.ndarray,
                            angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotates positional data by a given angle.

        :param Lat:  Latitude array in degrees
        :type: np.ndarray
        :param Long: Longitude array in degrees
        :type: np.ndarray
        :param angle:    Rotation angle in degrees.
        :type: float
        :return: Rotated X and Y coordinates
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        return (
            Lat * np.cos(angle * np.pi / 180) - Long * np.sin(angle * np.pi / 180),
            Lat * np.sin(angle * np.pi / 180) + Long * np.cos(angle * np.pi / 180)
        )

    def differential_rotation(self, Lat: np.ndarray) -> np.ndarray:
        """
        Calculates the differential rotation of the Sun based on latitude.
        :param Lat:  Latitude array in degrees
        :type: np.ndarray
        :return:
        """
        A = 14.713
        B = -2.396
        C = -1.787
        return A + B * np.sin(Lat * np.pi / 180) ** 2 + C * np.sin(Lat * np.pi / 180) ** 4

    def pozitional_angle(self, AZIMUTH: float, SOL_DEC: float, SOLAR_P: float) -> float:
        """
         Calculates the positional angle of the sun based on azimuth, solar declination, and solar P-angle.
        :param AZIMUTH: Azimuth angle.
        :type: float
        :param SOL_DEC:  Solar declination angle.
        :type: float
        :param SOLAR_P: Solar P-angle
        :type: float
        :return: Positional angle of the sun
        :rtype: float
        """
        q = -np.arcsin(np.tan(AZIMUTH * np.pi / 180) * np.tan(SOL_DEC * np.pi / 180)) * 180 / np.pi
        p = SOLAR_P + 360.0 if np.abs(SOLAR_P) > 30 else SOLAR_P
        return (p + q)

    def form_srstable_with_time_shift(self, processed_file: fits.HDUList) -> Table:
        """ Create table with AR info with SRSClient
        with correct NOAA coordinates for RATAN time difference

        :param processed_file:  calibrate RATAN fits
        :type: fits.HDUList
        :return: modified NOAA table with ARs
        :rtype: Table
        """

        SOLAR_R = processed_file[0].header['SOLAR_R'] * 1.01175
        SOLAR_B = processed_file[0].header['SOLAR_B']
        OBS_DATE = processed_file[0].header['DATE-OBS']
        OBS_TIME = processed_file[0].header['TIME-OBS']

        AZIMUTH = processed_file[0].header['AZIMUTH']
        SOL_DEC = processed_file[0].header['SOL_DEC']
        SOLAR_P = processed_file[0].header['SOLAR_P']
        angle = self.pozitional_angle(AZIMUTH, SOL_DEC, SOLAR_P)
        ratan_datetime = datetime.strptime(OBS_DATE + ' ' + OBS_TIME, '%Y/%m/%d %H:%M:%S.%f')
        ratan_datetime_str = ratan_datetime.strftime('%Y/%m/%d %H:%M')
        noaa_datetime = datetime.strptime(OBS_DATE, '%Y/%m/%d')
        diff_hours = int((ratan_datetime - noaa_datetime).total_seconds() / 3600)

        srs = SRSClient()
        srs_table = srs.get_data(TimeRange(OBS_DATE, OBS_DATE))
        srs_table = srs_table[srs_table['ID'] == 'I']
        len_tbl = len(srs_table)
        srs_table.add_column(Column(name='RatanTime', data=[ratan_datetime_str] * len_tbl))
        srs_table['Longitude'] = (
                srs_table['Longitude'] + self.differential_rotation(srs_table['Latitude']) * diff_hours / 24
        ).astype(int)

        srs_table['Latitude'], srs_table['Longitude'] = self.heliocentric_transform(
            srs_table['Latitude'],
            srs_table['Longitude'],
            SOLAR_R,
            SOLAR_B
        )

        srs_table['Latitude'], srs_table['Longitude'] = self.pozitional_rotation(
            srs_table['Latitude'],
            srs_table['Longitude'],
            angle
        )

        return srs_table

    def active_regions_search(self, srs: Table, x: np.ndarray, V: np.ndarray, mask: np.ndarray):
        """
         Searches for active regions in the solar spectrum data using wavelet denoising and peak detection
         and matching with NOAA active location.
        :param srs: table from SRSCLient with informations about the NOAA active regions.
        :param x: x coordinates
        :param V: Stocks vector V
        :param mask: masked interval
        :return: Table with active regions: V value and coordinate for center of AR
        :rtype: Table
        """
        x = x[mask]
        V = np.sum(V, axis=0)[mask]

        wavelet = 'sym6'  # Daubechies wavelet
        level = 4  # Level of decomposition
        denoised_data = wavelet_denoise(V, wavelet, level)
        height_threshold = lambda x: np.abs(np.median(x) + 0.1 * np.std(x))
        # Finding peaks in the denoised data
        peaks, _ = find_peaks(denoised_data, height=height_threshold(denoised_data))
        valleys, _ = find_peaks(-denoised_data, height=height_threshold(-denoised_data))
        extremums = np.concatenate((peaks, valleys))

        theoretical_latitudes = np.array(srs['Latitude'])
        experimental_latitudes = x[extremums]
        abs_diff = np.abs(experimental_latitudes[:, np.newaxis] - theoretical_latitudes)
        min_index = np.argmin(abs_diff, axis=1)
        closest_data = srs[min_index]
        closest_data['Latitude'] = experimental_latitudes

        original_indices = np.where(mask)[0]
        extremums_original = original_indices[extremums]
        closest_data.add_column(Column(name='Data Index', data=extremums_original, dtype=('i4')), index=3)
        closest_data.add_column(Column(name='Masked Index', data=extremums, dtype=('i4')), index=4)
        return closest_data

    def make_multigauss_fit(self, x: np.ndarray, y: np.ndarray,
                            peak_info: Table) -> Tuple[float, float, float]:
        """
        Fits multiple Gaussian functions to detected peaks.
        :param x: x values of the data.
        :type: np.ndarray
        :param y: y values of the data.
        :type: np.ndarray
        :param peak_info:  astropy Table with information about peaks.
        :return: width, value and coordinate for center of AR.
        :rtype: Tuple[float, float, float]
        """
        min_lat = np.min(peak_info['Latitude'])
        max_lat = np.max(peak_info['Latitude'])
        indexes = peak_info['Data Index']
        mask = (x >= min_lat - 50) & (x <= max_lat + 50)
        x_masked, y_masked = x[mask], y[mask]
        y_min = np.min(y_masked)
        ar_info = [[y[index] - y_min, x[index]] for index in indexes]
        widths = np.repeat(1, len(peak_info))
        initial_guesses = np.ravel(np.column_stack((ar_info, widths)))
        params, _ = leastsq(gaussian_mixture, initial_guesses, args=(x_masked, y_masked - y_min))
        return np.array(params), y_min, x_masked

    def gauss_analysis(self, x, scan_data, ar_info):
        """
        Analyzes active regions by fitting Gaussian functions and calculating associated metrics.
        :param x: x coordinates of the data (arcsec).
        :param scan_data:  scan with radio data
        :type: np.ndarray
        :param ar_info: Table with information about active regions.
        :type: Table
        :return: Table with ARs parameters from Radio data at each frequency
        :rtype: Table
        """
        ar_number = np.unique(ar_info['Number'])
        for noaa_ar in ar_number:
            region_info = ar_info[ar_info['Number'] == noaa_ar]
            indices = np.where(ar_info['Number'] == noaa_ar)[0]
            for index, elem in enumerate(scan_data):
                freq, I_data, V_data = elem
                gauss_params, y_min, x_range = self.make_multigauss_fit(x, I_data, region_info)
                gauss_params = gauss_params.reshape(-1, 3)
                gauss_params[:, 0] += y_min
                total_flux = np.sum(np.sqrt(2 * np.pi) * gauss_params[:, 0] * gauss_params[:, 2])
                for local_index, gaussian in zip(indices, gauss_params):
                    amplitude, mean, stddev = gaussian
                    ar_info['Amplitude'][local_index][index] = {'freq': freq, 'amplitude': amplitude}
                    ar_info['Mean'][local_index][index] = {'freq': freq, 'mean': mean}
                    ar_info['Sigma'][local_index][index] = {'freq': freq, 'sigma': stddev}
                    ar_info['FWHM'][local_index][index] = {'freq': freq, 'fwhm': 2 * np.sqrt(2 * np.log(2)) * stddev}
                    ar_info['Range'][local_index][index] = {'freq': freq, 'x_range': (np.min(x_range), np.max(x_range))}
                    ar_info['Flux'][local_index][index] = {'freq': freq,
                                                           'flux': np.sqrt(2 * np.pi) * amplitude * stddev}
                    ar_info['Total Flux'][local_index][index] = {'freq': freq, 'flux': total_flux}
        return ar_info

    def acquire_data(self, timerange: TimeRange) -> List[str]:
        """
        Get all available urls for predefined TimeRange
        :param timerange: timerange for scan searching
        :type: TimeRange
        :return: url list:
        :rtype:
        """
        scrapper = Scrapper(self.base_url, regex_pattern=self.regex_pattern,
                            condition=self.condition, filter=self.filter)
        return scrapper.form_fileslist(timerange)

    def get_scans(self, timerange):
        file_urls = self.acquire_data(timerange)
        column_types = {
            'Date': np.dtype('U10'),
            'Time': np.dtype('U8'),
            'Azimuth': np.dtype('i2'),
            'SOLAR_R': np.dtype('float64'),
            'N_shape': np.dtype('i4'),
            'CRPIX': np.dtype('float64'),
            'CDELT1': np.dtype('float64'),
            'Pozitional Angle': np.dtype('float64'),
            'SOLAR_B': np.dtype('float64'),
            'Frequency': np.dtype('O'),
            'Flux Eficiency ': np.dtype('O'),
            'I': np.dtype('O'),
            'V': np.dtype('O'),
        }

        table = Table(names=tuple(column_types.keys()), dtype=tuple(column_types.values()))
        for file_url in file_urls:
            hdul = fits.open(file_url)
            data = hdul[0].data

            CDELT1 = hdul[0].header['CDELT1']
            CRPIX = hdul[0].header['CRPIX1']
            SOLAR_R = hdul[0].header['SOLAR_R'] * 1.01175
            SOLAR_B = hdul[0].header['SOLAR_B']
            FREQ = hdul[1].data['FREQ']
            OBS_DATE = hdul[0].header['DATE-OBS']
            OBS_TIME = hdul[0].header['TIME-OBS']
            bad_freq = np.isin(FREQ, [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062])

            AZIMUTH = hdul[0].header['AZIMUTH']
            SOL_DEC = hdul[0].header['SOL_DEC']
            SOLAR_P = hdul[0].header['SOLAR_P']
            angle = self.pozitional_angle(AZIMUTH, SOL_DEC, SOLAR_P)
            N_shape = data.shape[2]
            x = np.linspace(
                - CRPIX * CDELT1,
                (N_shape - CRPIX) * CDELT1,
                num=N_shape
            )
            flux_eficiency = self.antenna_efficiency(FREQ, SOLAR_R)
            mask, I, V = self.calibrate_QSModel(x, data, SOLAR_R, FREQ, flux_eficiency)
            I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]
            table.add_row([
                OBS_DATE.replace('/', '-'),
                OBS_TIME,
                AZIMUTH,
                SOLAR_R,
                N_shape,
                CRPIX,
                CDELT1,
                angle,
                SOLAR_B,
                FREQ,
                dict(zip(FREQ, flux_eficiency)),
                dict(zip(FREQ, I)),
                dict(zip(FREQ, V))
            ])
        return table

    def form_data(self, file_urls):
        total_table = []
        for file_url in file_urls:
            hdul = fits.open(file_url)
            data = hdul[0].data

            CDELT1 = hdul[0].header['CDELT1']
            CRPIX = hdul[0].header['CRPIX1']
            SOLAR_R = hdul[0].header['SOLAR_R'] * 1.01175
            SOLAR_B = hdul[0].header['SOLAR_B']
            FREQ = hdul[1].data['FREQ']
            OBS_DATE = hdul[0].header['DATE-OBS']
            OBS_TIME = hdul[0].header['TIME-OBS']
            bad_freq = np.isin(FREQ, [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062])

            ratan_datetime = datetime.strptime(OBS_DATE + ' ' + OBS_TIME, '%Y/%m/%d %H:%M:%S.%f')
            noaa_datetime = datetime.strptime(OBS_DATE, '%Y/%m/%d')
            diff_hours = int((ratan_datetime - noaa_datetime).total_seconds() / 3600)

            AZIMUTH = hdul[0].header['AZIMUTH']
            SOL_DEC = hdul[0].header['SOL_DEC']
            SOLAR_P = hdul[0].header['SOLAR_P']
            angle = self.pozitional_angle(AZIMUTH, SOL_DEC, SOLAR_P)

            x = np.linspace(
                - CRPIX * CDELT1,
                (data.shape[2] - CRPIX) * CDELT1,
                num=data.shape[2]
            )

            flux_eficiency = self.antenna_efficiency(FREQ, SOLAR_R)
            mask, I, V = self.calibrate_QSModel(x, data, SOLAR_R, FREQ, flux_eficiency)
            I, V, FREQ = I[~bad_freq], V[~bad_freq], FREQ[~bad_freq]

            srs = SRSClient()
            srs_table = srs.get_data(TimeRange(OBS_DATE, OBS_DATE))
            srs_table = srs_table[srs_table['ID'] == 'I']
            srs_table['Longitude'] = (
                    srs_table['Longitude'] + self.differential_rotation(srs_table['Latitude']) * diff_hours / 24
            ).astype(int)

            srs_table['Latitude'], srs_table['Longitude'] = self.heliocentric_transform(
                srs_table['Latitude'],
                srs_table['Longitude'],
                SOLAR_R,
                SOLAR_B
            )

            srs_table['Latitude'], srs_table['Longitude'] = self.pozitional_rotation(
                srs_table['Latitude'],
                srs_table['Longitude'],
                angle
            )

            ar_info = self.active_regions_search(srs_table, x, V, mask)
            ar_amount = len(ar_info)
            ar_info.add_column(Column(name='Azimuth', data=[AZIMUTH] * ar_amount, dtype=('i2')), index=1)
            ar_info.add_column(
                Column(name='Amplitude', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(
                Column(name='Mean', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(
                Column(name='Sigma', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(
                Column(name='FWHM', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(
                Column(name='Flux', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))
            ar_info.add_column(
                Column(name='Total Flux', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)],
                       dtype=object))
            ar_info.add_column(
                Column(name='Range', data=[[{} for _ in range(len(FREQ))] for _ in range(ar_amount)], dtype=object))

            ratan_data = list(zip(FREQ, I, V))
            gauss_analysis_info = self.gauss_analysis(x, ratan_data, ar_info)
            total_table.append(gauss_analysis_info)
        return Table(vstack(total_table))

    def get_data(self, timerange):
        file_urls = self.acquire_data(timerange)
        return self.form_data(file_urls)
