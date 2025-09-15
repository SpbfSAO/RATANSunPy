from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from numpy import ndarray
from ratansunpy.client import RATANClient
from scipy.ndimage import binary_fill_holes


class ARHandler:
    def __init__(
            self,
            calibrated_data: fits.HDUList,
            bad_freq: Optional[List[float]] = None,
            window_size: int = 50,
            scrap_srs_table: bool = True,
            srs_table: Optional[Table] = None,
            srs_base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize the ARHandler for extracting and processing active regions (ARs).

        :param calibrated_data: Calibrated FITS data containing intensity (I), circular polarization (V), and frequency (FREQ).
        :param bad_freq: List of frequencies to exclude (default: predefined bad frequencies).
        :param window_size: Size of the window around AR centers (default: 100 pixels).
        :param scrap_srs_table: Whether to scrape the SRS table if not provided (default: True).
        :param srs_table: Preloaded SRS table (optional).
        :param srs_base_url: Base URL for scraping the SRS table (optional).
        """

        assert isinstance(
            calibrated_data, fits.HDUList), "Input data should be hdu list type"
        if bad_freq is None:
            bad_freq = [15.0938, 15.2812, 15.4688, 15.6562,
                        15.8438, 16.0312, 16.2188, 16.4062]
        self.bad_freq = bad_freq

        # Extract header and data
        self.CDELT1 = calibrated_data[0].header['CDELT1']
        self.CRPIX = calibrated_data[0].header['CRPIX1']
        FREQ = calibrated_data[3].data
        bad_freq_mask = np.isin(FREQ, bad_freq)
        self.DATE_OBS = calibrated_data[0].header['DATE-OBS']
        self.TIME_OBS = calibrated_data[0].header['TIME-OBS']
        self.AZIMUTH = calibrated_data[0].header['AZIMUTH']
        self.SOLAR_R = calibrated_data[0].header['SOLAR_R']
        self.SOLAR_B = calibrated_data[0].header['SOLAR_B']
        self.SOL_DEC = calibrated_data[0].header['SOL_DEC']
        self.ANGLE = calibrated_data[0].header['ANGLE']
        self.I = calibrated_data[1].data[~bad_freq_mask]
        self.V = calibrated_data[2].data[~bad_freq_mask]
        self.FREQ = FREQ[~bad_freq_mask]
        self.mask = calibrated_data[4].data.astype(bool)
        # Solar x-coordinates
        self.solar_x = np.linspace(
            -self.CRPIX * self.CDELT1,
            (self.V.shape[1] - self.CRPIX) * self.CDELT1,
            num=self.V.shape[1]
        )
        self.window_size = window_size

        # Load or scrape SRS table
        if srs_table is None and scrap_srs_table:
            self.srs_table = RATANClient().form_srstable_with_time_shift(
                calibrated_data, base_url=srs_base_url)
        else:
            self.srs_table = srs_table

    def extract_ar_data_with_window(
            self,
            latitude: float,
            window_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract a patch from the full scan with ±window_size around the given latitude.

        :param latitude: Latitude of the AR center.
        :param window_size: Size of the window (default: self.window_size).
        :return: Extracted spectrum data as a numpy array.
        """
        if window_size is None:
            window_size = self.window_size

        len_x = len(self.solar_x)

        center_index = np.argmin(np.abs(self.solar_x - latitude))
        left_index = max(0, center_index - window_size)
        right_index = min(len_x, center_index + window_size + 1)

        # Handle padding if the window exceeds the data boundaries
        pad_left = max(0, window_size - center_index)
        pad_right = max(0, (center_index + window_size) - len_x)

        # Extract data
        nfreq = self.I.shape[0]
        spectrum_data = np.zeros((2, nfreq, 2 * window_size + 1))
        spectrum_data[0, :, pad_left:2 * window_size + 1 -
                      pad_right] = self.I[:, left_index:right_index]
        spectrum_data[1, :, pad_left:2 * window_size + 1 -
                      pad_right] = self.V[:, left_index:right_index]

        return spectrum_data

    def vis_ar_2d(self,
                  spectrum_data: np.ndarray,
                  value: str = "I",
                  title: str = "AR Spectrum"):
        plt.figure(figsize=(10, 6))
        idx = 0 if value == "I" else 1
        plt.matshow(spectrum_data[idx])
        plt.show()
        return plt.gcf()

    @staticmethod
    def identify_and_replace_outliers(spectrum_data: np.ndarray, threshold_multiplier: float = 2.0) -> ndarray:
        """
       Identify and replace outliers in a spectrum.

       :param spectrum: 1D array representing the spectrum.
       :param threshold_multiplier: Multiplier for the 99th percentile to define outliers (default: 2.0).
       :return: Spectrum with outliers replaced by the average of neighbors.
       """

        if not isinstance(spectrum_data, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        if spectrum_data.ndim == 3:
            data = spectrum_data[0]
        else:
            data = spectrum_data
        percentile_99 = np.percentile(data, 99)

        # Define the outlier threshold
        outlier_threshold = threshold_multiplier * percentile_99

        # Identify outliers
        outlier_indices = np.where(data > outlier_threshold)[0]

        # Replace outliers with the average of neighbors
        for idx in outlier_indices:
            # Get neighboring values (avoid boundary issues)
            left = data[idx - 1] if idx > 0 else data[idx + 1]
            right = data[idx + 1] if idx < len(data) - 1 else data[idx - 1]
            data[idx] = (left + right) / 2

        if spectrum_data.ndim == 3:
            spectrum_data[0] = data
        else:
            spectrum_data = data

        return spectrum_data

    @staticmethod
    def compute_ar_mask(spectrum_data: np.ndarray,
                        dec_coeff: float = 2.5) -> np.ndarray:
        """
        Smooth the spectrum data with a 2D Gaussian and compute a mask. Filter out bad data points.

        :param spectrum_data: Extracted spectrum data.
        :param perc: Percentile to half
        :return: Boolean mask indicating significant regions.
        """
        if not isinstance(spectrum_data, np.ndarray):
            raise ValueError("Input must be a NumPy array")
        if spectrum_data.ndim == 3:
            spectrum_data = spectrum_data[0]

        top_perc = np.percentile(spectrum_data, 99)
        mask = spectrum_data > top_perc / dec_coeff
        mask = binary_fill_holes(mask)

        return mask

    def compute_ar_stats(self,
                         spectrum_data: np.ndarray,
                         mask: np.ndarray = None,
                         threshold_multiplier: float = 2.5,
                         ax_data: int = 0,
                         ax_along: int = 1) -> dict:
        """
        Compute basic statistics about the spectrum.

        :param spectrum_data: Extracted spectrum data for one cpmponent (ex I).
        :return: Dictionary containing statistics (mean, std, min, max).
        """
        spectrum_data = self.identify_and_replace_outliers(spectrum_data,
                                                           threshold_multiplier=threshold_multiplier)
        if mask is not None:
            spectrum_data = spectrum_data * mask
            spectrum_data = np.ma.masked_equal(spectrum_data, 0)
        if ax_data is not None:
            data = spectrum_data[ax_data]
        else:
            data = spectrum_data
        return {
            'mean': np.mean(data, axis=ax_along).filled(np.nan),
            'std': np.std(data, axis=ax_along).filled(np.nan),
            'min': np.min(data, axis=ax_along).filled(np.nan),
            'max': np.max(data, axis=ax_along).filled(np.nan),
            'sum': np.sum(data, axis=1).filled(np.nan),
        }

    def get_lat_intersections(self,
                              latitude: float,
                              window_size: Optional[int] = None):
        """
        Return nearby active regions by latitude within a window.

        Args
            latitude : float
                Central latitude to compare against.
            window_size : Optional[int]
                Range above and below the latitude to search in arcsec.

        Returns
            str
                Formatted string of AR numbers and latitudes (e.g., 'AR12345:-10.3'),
                or 'NONE' if no matches.
        """
        if window_size is None:
            window_size = self.window_size

        area_lat_dict = dict(
            zip(self.srs_table['Number'], self.srs_table['Latitude']))

        min_lat = latitude - window_size
        max_lat = latitude + window_size
        intercepts = {}

        for ar, lat in area_lat_dict.items():
            if (min_lat <= lat <= max_lat) and lat != latitude:
                intercepts[ar] = lat

        if not intercepts:
            return "NONE"
        return ",".join(f"AR{ar}:{lat:.2f}" for ar, lat in sorted(intercepts.items()))

    def get_mag_type(self,
                     latitude: float,
                     ar_number: str,):
        """
        Return the magnetic type of an active region by number and latitude.

        Args
            latitude : float
                Latitude of the active region.
            ar_number : str
                Active region number.

        Returns
            str
                Magnetic classification (e.g., 'Beta'), or 'NONE' if not found.
        """
        mask = (self.srs_table['Number'] == ar_number) & (
            self.srs_table['Latitude'] == latitude)
        if any(mask):
            return str(self.srs_table['Mag Type'][mask][0])
        else:
            return "NONE"

    def get_mcintosh(self,
                     latitude: float,
                     ar_number: str,):
        """
        Return mcintosh classification of an active region by number and latitude.

        Args
            latitude : float
                Latitude of the active region.
            ar_number : str
                Active region number.

        Returns
            str
                mcintosh classification (e.g., 'Hsx'), or 'NONE' if not found.
        """
        mask = (self.srs_table['Number'] == ar_number) & (
            self.srs_table['Latitude'] == latitude)
        if any(mask):
            return str(self.srs_table['Z'][mask][0])
        else:
            return "NONE"

    def process_one_region(
            self,
            latitude: float,
            ar_number: str,
            window_size: Optional[int] = None,
            threshold_multiplier: float = 2.5,
    ) -> tuple[fits.HDUList, str]:

        # Extract AR data
        spectrum_data = self.extract_ar_data_with_window(latitude, window_size)

        # Compute AR mask and statistics
        ar_mask = np.expand_dims(self.compute_ar_mask(
            spectrum_data).astype('float'), axis=0)
        spectrum_data = np.concatenate((spectrum_data, ar_mask), axis=0)
        ar_stats = self.compute_ar_stats(spectrum_data,
                                         mask=ar_mask,
                                         threshold_multiplier=threshold_multiplier,
                                         ax_data=0,
                                         ax_along=1)

        # Create FITS HDU
        primary_hdu = fits.PrimaryHDU(spectrum_data)
        primary_hdu.header['AR_NUM'] = ar_number
        primary_hdu.header['LATITUDE'] = latitude
        primary_hdu.header['DATE-OBS'] = self.DATE_OBS
        primary_hdu.header['TIME-OBS'] = self.TIME_OBS
        primary_hdu.header['CDELT1'] = self.CDELT1
        primary_hdu.header['CRPIX '] = self.CRPIX

        primary_hdu.header['AZIMUTH'] = self.AZIMUTH
        primary_hdu.header['SOLAR_R'] = self.SOLAR_R
        primary_hdu.header['SOLAR_B'] = self.SOLAR_B
        primary_hdu.header['SOL_DEC'] = self.SOL_DEC
        primary_hdu.header['ANGLE'] = self.ANGLE

        primary_hdu.header['MAG_TYPE'] = self.get_mag_type(latitude=latitude,
                                                           ar_number=ar_number)
        primary_hdu.header['MCINTOSH'] = self.get_mcintosh(latitude=latitude,
                                                           ar_number=ar_number)
        primary_hdu.header['LAT_INTR'] = self.get_lat_intersections(latitude=latitude,
                                                                                  window_size=window_size)

        hdu_list = [primary_hdu]
        for key, value in ar_stats.items():
            hdu_list.append(fits.ImageHDU(
                data=value.astype('float32'), name=key))
        hdu_list.append(fits.ImageHDU(
            data=self.FREQ.astype('float32'), name='FREQ'))
        ar_hdulist = fits.HDUList(hdu_list)
        ar_hdulist.verify('fix')
        # Combine date and time
        datetime_obj = datetime.strptime(
            f"{self.DATE_OBS} {self.TIME_OBS}", "%Y/%m/%d %H:%M:%S.%f")

        # Format the result as required
        timestamp = datetime_obj.strftime("%Y%m%d_%H%M%S")

        # Save to FITS file
        filename = f"{timestamp}_AR{ar_number}_{self.AZIMUTH}.fits"

        return ar_hdulist, filename

    def extract_ars_from_scan(self, save_path=None) -> list:
        """
        Extract all ARs, process them, and save to FITS files.

        :return: List of tuples containing AR filenames and corresponding FITS HDUs.
        """
        ar_data = []
        for row in self.srs_table:
            ar_number = row['Number']
            latitude = row['Latitude']
            ar_hdul, filename = self.process_one_region(
                latitude=latitude, ar_number=ar_number)
            ar_data.append((ar_hdul, filename))
            if save_path:
                ar_hdul.writeto(Path(save_path) / filename, overwrite=True)

        return ar_data
