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
from ratansunpy.client.ar_client import ARClient

import warnings
warnings.filterwarnings("ignore")


AR_DATA_DIR = os.path.join(os.getcwd(), 'data', 'AR_data') # Folder where to save extracted AR fits
TIMERANGE = TimeRange('2024-12-01', '2024-12-03') # Period to parse


def main():
    """Downloads Active Regions fits from SAO server."""
    arclient = ARClient()    
    arclient.download_data(timerange=TIMERANGE, save_to=AR_DATA_DIR)


if __name__ == '__main__':
    main()
