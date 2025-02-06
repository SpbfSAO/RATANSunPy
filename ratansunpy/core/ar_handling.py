from astropy.io import fits
from astropy.table import Table
from typing import Optional, List, Tuple
from ratansunpy.client import SRSClient, RATANClient
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.stats import zscore
import matplotlib.pyplot as plt


from ratansunpy.scrapper import Scrapper
from ratansunpy.time import TimeRange


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

        assert isinstance(calibrated_data, fits.HDUList), "Input data should be hdu list type"
        if bad_freq is None:
            bad_freq = [15.0938, 15.2812, 15.4688, 15.6562, 15.8438, 16.0312, 16.2188, 16.4062]
        self.bad_freq = bad_freq

            # Extract header and data
        self.CDELT1 = calibrated_data[0].header['CDELT1']
        self.CRPIX = calibrated_data[0].header['CRPIX1']
        FREQ = calibrated_data[3].data
        bad_freq_mask = np.isin(FREQ, bad_freq)
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
            self.srs_table = RATANClient().form_srstable_with_time_shift(calibrated_data, base_url=srs_base_url)
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
        right_index = min(len_x, center_index + window_size+1)

        # Handle padding if the window exceeds the data boundaries
        pad_left = max(0, window_size - center_index)
        pad_right = max(0, (center_index + window_size) - len_x)


        # Extract data
        nfreq = self.I.shape[0]
        spectrum_data = np.zeros((2, nfreq, 2 * window_size + 1))
        spectrum_data[0, :, pad_left:2 * window_size + 1 - pad_right] = self.I[:, left_index:right_index]
        spectrum_data[1, :, pad_left:2 * window_size + 1 - pad_right] = self.V[:, left_index:right_index]

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

    def check_bad_data(self, spectrum_data: np.ndarray) -> tuple:
        """
        Check data quality by identifying outliers and noisy regions.

        :param spectrum_data: Extracted spectrum data.
        :return: Boolean mask indicating good data points.
        """
        # Use z-score to identify outliers
        res = np.percentile(spectrum_data[0], [99, 100])
        return (res[0], res[1])

    def compute_ar_mask(self,
                        spectrum_data: np.ndarray,
                        dec_coeff: float = 2.5) -> np.ndarray:
        """
        Smooth the spectrum data with a 2D Gaussian and compute a mask. Filter out bad data points.

        :param spectrum_data: Extracted spectrum data.
        :param perc: Percentile to half
        :return: Boolean mask indicating significant regions.
        """
        top_perc = np.percentile(spectrum_data[0], 99)
        mask = spectrum_data[0] > top_perc/dec_coeff
        mask = binary_fill_holes(mask)

        return mask

    def compute_ar_stats(self, spectrum_data: np.ndarray, mask: np.ndarray = None) -> dict:
        """
        Compute basic statistics about the spectrum.

        :param spectrum_data: Extracted spectrum data.
        :return: Dictionary containing statistics (mean, std, min, max).
        """
        if mask is not None:
            spectrum_data = spectrum_data*mask
            spectrum_data = np.ma.masked_equal(spectrum_data, 0)
        return {
            'mean': np.mean(spectrum_data, axis=1),
            'std': np.std(spectrum_data, axis=1),
            'min': np.min(spectrum_data, axis=1),
            'max': np.max(spectrum_data, axis=1),
            'sum': np.sum(spectrum_data, axis=1),
        }

    def extract_ars_from_scan(self) -> List[Tuple[str, fits.HDUList]]:
        """
        Extract all ARs, process them, and save to FITS files.

        :return: List of tuples containing AR filenames and corresponding FITS HDUs.
        """
        ar_files = []
        for row in self.srs_table:
            ar_number = row['Number']
            latitude = row['Latitude']

            # Extract AR data
            spectrum_data = self.extract_ar_data_with_window(latitude)

            # Check data quality
            good_data_mask = self.check_bad_data(spectrum_data)

            # Compute AR mask and statistics
            ar_mask = self.compute_ar_mask(spectrum_data)
            ar_stats = self.compute_ar_stats(spectrum_data, ar_mask)

            # Create FITS HDU
            primary_hdu = fits.PrimaryHDU(spectrum_data)
            primary_hdu.header['AR_NUM'] = ar_number
            primary_hdu.header['LATITUDE'] = latitude
            for key, value in ar_stats.items():
                primary_hdu.header[key.upper()] = value

            # Save to FITS file
            filename = f"{row['Timestamp']}_sun+0_AR{ar_number}.fits"
            ar_files.append((filename, fits.HDUList([primary_hdu])))

        return ar_files
