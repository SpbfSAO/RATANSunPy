import re
from datetime import datetime
from ftplib import FTP
from typing import List, Optional, Callable, Any
from urllib.parse import urlsplit
import requests
from astropy.time import Time
from dateutil.relativedelta import relativedelta

from ratansunpy.time import TimeRange

TIME_REGEX = {'%Y': r'\d{4}', '%y': r'\d{2}',
              '%b': '[A-Z][a-z]{2}', '%m': r'\d{2}',
              '%d': r'\d{2}', '%j': r'\d{3}',
              '%H': r'\d{2}',
              '%M': r'\d{2}',
              '%S': r'\d{2}'}

__all__ = ['Scrapper']

class Scrapper:
    def __init__(
            self,
            baseurl: str,
            regex_pattern: Optional[str] = None,
            condition: Optional[Callable[[str, str, str], str]] = None,
            filter: Optional[Callable[[str], bool]] = None,
            **kwargs: Any
    ) -> None:
        """
        Initialize the Scrapper object with base URL and optional parameters.

        :param baseurl: The base URL to scrape data from.
        :param regex_pattern: Optional regex pattern to match filenames.
        :param condition: Optional callable to generate dates based on extracted data.
        :param filter: Optional filter to apply to the extracted data.
        """

        self.baseurl = baseurl
        self.domain = f"{urlsplit(self.baseurl).scheme}://{urlsplit(self.baseurl).netloc}/"
        self.filter = filter
        self.regex_pattern = regex_pattern
        self.condition = condition

    @staticmethod
    def smallest_significant_pattern(pattern: str) -> Optional[relativedelta]:
        """
        Determine the smallest significant pattern (e.g., seconds, minutes, days) in the given pattern.
        Some of them are here: https://fits.gsfc.nasa.gov/iso-time.html

        :param pattern: The pattern string.
        :return: The smallest significant `relativedelta` object, or None if not found.
        """
        try:
            if any(second in pattern for second in ['%S']):
                return relativedelta(seconds=1)
            elif any(minute in pattern for minute in ['%M']):
                return relativedelta(minutes=1)
            elif any(hour in pattern for hour in ['%H']):
                return relativedelta(hours=1)
            elif any(day in pattern for day in ['%d', '%j']):
                return relativedelta(days=1)
            elif any(month in pattern for month in ['%m', '%b']):
                return relativedelta(months=1)
            if any(year in pattern for year in ['%y', '%Y']):
                return relativedelta(years=1)
            else:
                return None
        except Exception:
            raise

    @staticmethod
    def floor_datetime(date: Time, timestep: relativedelta) -> datetime:
        """
        Floor the given datetime to the nearest significant time unit.

        :param date: The `Time` object to floor.
        :param timestep: The `relativedelta` object representing the smallest significant time unit.
        :return: The floored `datetime` object.
        """
        date = date.to_datetime()
        if timestep.years > 0:
            return date.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        elif timestep.months > 0:
            return date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        elif timestep.days > 0:
            return date.replace(hour=0, minute=0, second=0, microsecond=0)
        elif timestep.hours > 0:
            return date.replace(minute=0, second=0, microsecond=0)
        elif timestep.minutes > 0:
            return date.replace(second=0, microsecond=0)
        return date

    def range(self, timerange: TimeRange) -> List[str]:
        """
        Generate a list of directories within the time range based on the smallest significant pattern.

       :param timerange: The `TimeRange` object representing the time range.
       :return: A list of directory paths.
       """
        filepath_pattern = '/'.join(self.baseurl.split('/')[:-1]) + '/'
        smallest_step = self.smallest_significant_pattern(filepath_pattern)
        if smallest_step is None:
            return [filepath_pattern]
        directories = []
        current_date = self.floor_datetime(timerange.start, smallest_step)
        end_date = self.floor_datetime(timerange.end, smallest_step) + smallest_step
        while current_date < end_date:
            directories.append(current_date.strftime(filepath_pattern))
            current_date += smallest_step
        return directories

    def extract_date_from_url(self, url):
        """
        Extract date from a given URL based on the base URL's pattern.

        :param url: The URL string.
        :return: The extracted `Time` object.
        """
        def url2list(text):
            return re.sub(r'\.|_', '/', text).split('/')

        pattern_parts = url2list(self.baseurl)
        url_parts = url2list(url)
        time_formats = ['%Y', '%y', '%b', '%B', '%m', '%d', '%j',
                        '%H', '%M', '%S']
        extracted_date, extracted_pattern = [], []
        for p_elem, u_elem in zip(pattern_parts, url_parts):
            present_formats = [x for x in time_formats if x in p_elem]
            part_to_remove = re.split('%.', p_elem)

            for candidate_to_remove in part_to_remove:
                if candidate_to_remove:
                    u_elem = u_elem.replace(candidate_to_remove, '', 1)
                    p_elem = p_elem.replace(candidate_to_remove, '', 1)

            extracted_date.append(u_elem)
            extracted_pattern.append(p_elem)
            time_formats = [fmt for fmt in time_formats if fmt not in present_formats]

        # Find the index of the fullest pattern
        fullest_pattern_index = extracted_pattern.index(max(extracted_pattern, key=len))
        # Find the corresponding date part
        date_part = extracted_date[fullest_pattern_index]
        return Time.strptime(date_part, extracted_pattern[fullest_pattern_index])

    def valid_date_from_url(self, url: str) -> bool:
        """
        Validate if a given URL's date matches the expected pattern from the base URL.

        :param url: The URL string to validate.
        :return: True if the URL's date matches the pattern, False otherwise.
        """
        pattern = self.baseurl
        # Replace datetime formats in the pattern string with their corresponding regex patterns
        for time_format, regex in TIME_REGEX.items():
            pattern = pattern.replace(time_format, regex)
        # Create a regex pattern object
        pattern_obj = re.compile(pattern)
        # Check if the URL matches the pattern
        return pattern_obj.fullmatch(url) is not None

    def check_date_in_timerange_from_url(self,
                                         url: str,
                                         timerange: TimeRange) -> bool:
        """
        Check if the date extracted from a URL is within the given time range.

        :param url: The URL string.
        :param timerange: The `TimeRange` object representing the time range.
        :return: True if the date is within the range, False otherwise.
        """
        file_date = self.extract_date_from_url(url).to_datetime()
        #smallest_pattern = self.smallest_significant_pattern(self.baseurl)
        file_range = TimeRange(file_date, file_date)
        return timerange.have_intersection(file_range)

    def check_date_in_timerange_from_file_date(self,
                                               file_date: str,
                                               timerange: TimeRange) -> bool:
        """
        Check if a given file date is within the specified time range.

        :param file_date: The file date as a string (format: "%Y-%m-%d").
        :param timerange: The `TimeRange` object representing the time range.
        :return: True if the date is within the range, False otherwise.
        """
        file_date = datetime.strptime(file_date, "%Y-%m-%d")
        #smallest_pattern = self.smallest_significant_pattern(self.baseurl)
        file_range = TimeRange(file_date, file_date)
        return timerange.have_intersection(file_range)

    def ftpfiles(self, timerange: TimeRange) -> List[str]:
        """
        Retrieve a list of files from an FTP server within the specified time range.

        :param timerange: The `TimeRange` object representing the time range.
        :return: A list of file URLs.
        """
        directories = self.range(timerange)
        file_urls = []
        ftpurl = urlsplit(directories[0]).netloc
        with FTP(ftpurl, user="anonymous", passwd="soleil@package") as ftp:
            for current_directory in directories:
                try:
                    ftp.cwd(urlsplit(current_directory).path)
                except Exception as e:
                    print(f'FTP CWD tried: {e}')
                    continue
                for file_name in ftp.nlst():
                    file_path = current_directory + file_name
                    if self.check_date_in_timerange_from_url(file_path, timerange):
                        file_urls.append(file_path)
        return file_urls

    def httpfiles(self, timerange: TimeRange) -> List[str]:
        """
        Retrieve a list of files from an HTTP server within the specified time range.

        :param timerange: The `TimeRange` object representing the time range.
        :return: A list of file URLs.
        """
        directories = self.range(timerange)
        file_urls = []
        for current_directory in directories:
            directory_parts = current_directory.split('/')
            year = directory_parts[-3]
            month = directory_parts[-2]
            try:
                page = requests.get(current_directory)
                page.raise_for_status()
            except (requests.exceptions.RequestException, ConnectionResetError) as err:
                continue
            for match in re.findall(fr'href="{self.regex_pattern}"', page.text):
                relative_path, date_text = match
                date = self.condition(year, month,
                                      date_text) if self.condition else f'{date_text[:-4]}-{date_text[-4:-2]}-{date_text[-2:]}'
                url = current_directory + relative_path
                if self.check_date_in_timerange_from_file_date(date, timerange):
                    file_urls.append(url)
        return file_urls

    def form_fileslist(self, timerange: TimeRange) -> List[str]:
        """
        Retrieve a list of files from an HTTP or FTP server within the specified time range.

        :param timerange: The `TimeRange` object representing the time range.
        :type timerange: TimeRange
        :return: A list of file URLs.

        :Example: usage example based on SWPC Solar Region Summary (FTP server)

        >>> base_url_SRS = r'ftp://ftp.ngdc.noaa.gov/STP/swpc_products/daily_reports/solar_region_summaries/%Y/%m/%Y%m%dSRS.txt'
        >>> scraper = Scrapper(base_url_SRS)
        >>> t = TimeRange('2021-10-12', '2021-10-12')
        >>> print(t)
        (2021-10-12 00:00:00, 2021-10-12 00:00:00)

        >>> for url in scraper.form_fileslist(t):
        >>>     print(f'SRS url: {url}')
        SRS url: ftp://ftp.ngdc.noaa.gov/STP/swpc_products/daily_reports/solar_region_summaries/2021/10/20211012SRS.txt

        :Example: usage example based on RATAN (HTTP server)

        >>> if int(year) < 2010 or (int(year) == 2010 and int(month) < 5):
        >>>     return f'{year[:2]}{date_match[:-4]}-{date_match[-4:-2]}-{date_match[-2:]}'
        >>> else:
        >>>    f'{date_match[:-4]}-{date_match[-4:-2]}-{date_match[-2:]}'
        >>> base_url_RATAN = 'http://spbf.sao.ru/data/ratan/%Y/%m/%Y%m%d_%H%M%S_sun+0_out.fits'
        >>> regex_pattern_RATAN = '((\d{6,8})[^0-9].*[^0-9]0_out.fits)'
        >>> scraper = Scrapper(base_url_RATAN, regex_pattern=regex_pattern_RATAN, condition=build_date)
        >>> t = TimeRange('2010-01-13', '2010-01-13')
        >>> for url in scraper.form_fileslist(t):
        >>>     print(f'RATAN url: {url}')
        RATAN url: http://spbf.sao.ru/data/ratan/2010/01/100113sun0_out.fits
        """
        # SWPC SRS, for example
        if urlsplit(self.baseurl).scheme == 'ftp':
            return self.ftpfiles(timerange)
        # RATAN, for example
        if urlsplit(self.baseurl).scheme in ['http', 'https']:
            return self.httpfiles(timerange)
