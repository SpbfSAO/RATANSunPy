
from pathlib import Path
from urllib.request import urlopen

from astropy.io import fits

from ratansunpy.scrapper import Scrapper
from ratansunpy.time import TimeRange
from ratansunpy.utils import *

import os
from urllib.request import urlopen

import tempfile
import sys


from astropy.io import fits
from tqdm import tqdm

from ratansunpy.core.ar_handling import ARHandler
from ratansunpy.time import TimeRange

from ratansunpy.utils import logger
import warnings
warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
from .client import BaseClient
from .client import RATANClient

logger = logger.get_logger(__name__)



class ARClient(BaseClient):
    
    def __init__(self, base_url=None, output_dir=os.getcwd()):
        self.main_url = 'http://spbf.sao.ru/data/solar_data/AR_data/%Y/%Y%m%d_%H%M%S_*.fits'
        self.base_url = base_url if base_url else self.main_url
        self.regex_pattern = r'(\d{8}_\d{6}_AR\d{4}_-?\d+\.\d+\.fits)'
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        

    def acquire_data(self, timerange: TimeRange) -> list[str]:
        """
        Retrieves a list of URLs of FITS files for the given time range.
        """
        scrapper = Scrapper(self.base_url, regex_pattern=self.regex_pattern)
        try:
            file_urls = scrapper.form_fileslist(timerange)
        except Exception as e:
            raise RuntimeError(f"Failed to get URLs: {e}")

        return file_urls


    def download_data(self, timerange: TimeRange, save_to: str = None) -> list[str]:
        """
        Downloads FITS files from the given URLs and saves them to the specified directory.
        Returns a list of paths to the saved files.
        """
        save_dir = save_to if save_to else self.output_dir
        os.makedirs(save_dir, exist_ok=True)
        file_urls = self.acquire_data(timerange)
        filepaths = []

        for url in file_urls:
            try:
                filename = os.path.basename(url)
                filepath = os.path.join(save_dir, filename)
                if not os.path.exists(filepath):
                    with urlopen(url) as response:
                        with open(filepath, 'wb') as out_file:
                            out_file.write(response.read())
                filepaths.append(filepath)
            except Exception as e:
                raise RuntimeError(f"Failed to download {url}: {e}")

        return filepaths


    def form_data(file_urls):
        raise NotImplementedError("The function 'form_data' is not implemented yet.")


    def get_data(self, timerange):
        raise NotImplementedError("The method 'get_data' is not implemented yet.")
