"""
Example on how to gather AR data.

Must have corresponding SRS and RATAN data to proccess and extract AR fits.
In this branch historic SRS data (up to 22.03.2025) can be found in ./data/SRS_data.

Current setting parse data for the period 01.12.2024 - 31.12.2024.
"""

import os
import sys

from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning
from tqdm import tqdm

from ratansunpy.client import RATANClient
from ratansunpy.core.ar_handling import ARHandler
from ratansunpy.time import TimeRange

import warnings
warnings.filterwarnings("ignore")


# Settings
SRS_DATA = os.path.join(os.getcwd(), "data", 'SRS_data', '%Y_SRS') # Folder path for SRS Data 
RATAN_DATA = os.path.join(os.getcwd(), 'data', 'RATAN_data', '2024') # Folder where to save processed RATAN fits
AR_DATA = os.path.join(os.getcwd(), 'data', 'AR_data') # Folder where to save extracted AR fits
TIMERANGE = TimeRange('2024-12-01', '2024-12-31') # Period to parse


def get_ratan_data(timerange: TimeRange, save_to: str) -> str:
    """
    Downloads RATAN-600 fits data for the specified period of time.

    Args:
        timerange: TimeRange - time range to fetch data for
        save_to: str - output folder

    Returns:
        str: Path to folder with saved RATAN-600 processed FITS files
    """
    os.makedirs(save_to, exist_ok=True)
    ratan_client = RATANClient()
    urls = ratan_client.acquire_data(timerange)

    for url in tqdm(urls,
                    desc="Downloading and processing RATAN-600 FITS",
                    unit="file",
                    file=sys.stdout,
                    ncols=100):
        try:
            ratan_client.process_fits_data(
                url,
                save_path=save_to,
                save_with_original=False
            )
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue

    return save_to


def get_ar_data(ratan_data: str, srs_data: str, save_to: str) -> None:
    """
    Extract processed ARs from RATAN-600 and SRS data

    Args:
        ratan_data: str - path to folder with RATAN processed fits files
        srs_data: str - path to folder with SRS data
        save_to: str - output folder
    """
    os.makedirs(save_to, exist_ok=True)
    fits_files = [f for f in os.listdir(ratan_data) if f.endswith('.fits')]

    for fits_file in tqdm(fits_files, 
                          desc="Extractig AR FITS from RATAN-600 and SRS Data",
                          unit="file",
                          file=sys.stdout,
                          ncols=100):
        try:
            processed_hdul = fits.open(os.path.join(ratan_data, fits_file))
            srs_base_url = os.path.join(srs_data, '%Y%m%dSRS.txt')

            handler = ARHandler(processed_hdul, srs_base_url=srs_base_url)
            
            
            handler.extract_ars_from_scan(save_to)
        
        except Exception as e:
            print(f"Error processing {fits_file}: {e}")
            continue


def main():
    """Downloads and process RATAN fits and extracts Active Regions fits."""
    ratan_fits_dir = get_ratan_data(
        timerange=TIMERANGE,
        save_to=RATAN_DATA
    )

    get_ar_data(
        ratan_data=ratan_fits_dir,
        srs_data=SRS_DATA,
        save_to=AR_DATA
    )

if __name__ == '__main__':
    main()
