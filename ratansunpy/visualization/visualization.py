import matplotlib.pyplot as plt
import matplotlib.colors as colors
import ratansunpy.visualization.colormaps as cm
from ratansunpy.client import RATANClient
from typing import Optional, Union
from astropy.io import fits
import numpy as np

def plot_ratan_fits_data(
                        pr_data: Union[str, fits.hdu.hdulist.HDUList],
                        is_calibrated: bool = False,
                        plot_V: Optional[bool] = True,
                        plot_I: Optional[bool] = True) -> None: 
    """
    Plot RATAN-600 FITS data for Stokes I and/or V parameters.
    
    :param pr_data: Path to FITS file, URL, or FITS HDUList with the RATAN-600 data.
    :type pr_data: Union[str, fits.hdu.hdulist.HDUList]
    :param is_calibrated: Indicates whether the data is already calibrated. Defaults to False.
    :type is_calibrated: bool
    :param plot_V: Whether to plot Stokes V data. Defaults to True.
    :type plot_V: Optional[bool]
    :param plot_I: Whether to plot Stokes I data. Defaults to True.
    :type plot_I: Optional[bool]
    :return: None
    :rtype: None
    :raises ValueError: If both plot_I and plot_V are set to False.
    """
    if not plot_I and not plot_V:
        raise ValueError("At least one of plot_I or plot_V must be True.")

    if isinstance(pr_data, str):
        ratan_client = RATANClient()
        raw, processed = ratan_client.process_fits_data(pr_data,
                                                    save_path=None,
                                                    save_with_original=False)
        ratan_file = processed if is_calibrated else raw
    elif isinstance(pr_data, fits.hdu.hdulist.HDUList):
        ratan_file = pr_data
        is_calibrated = True if ratan_file[0].data is None else False
    else:
        NotImplementedError

    SOLAR_R = ratan_file[0].header['SOLAR_R']
    CDELT1 = ratan_file[0].header['CDELT1']
    CRPIX1 = ratan_file[0].header['CRPIX1']
    DATE = ratan_file[0].header['DATE-OBS']

    if is_calibrated: 
        I, V, FREQ = ratan_file[1].data, ratan_file[2].data, ratan_file[3].data
    else: 
        I, V, FREQ = ratan_file[0].data[:, 0, :], ratan_file[0].data[:, 1, :], ratan_file[1].data['FREQ']

    x = np.linspace(
        - CRPIX1 * CDELT1,
        (I.shape[1] - CRPIX1) * CDELT1,
        num=I.shape[1]
    )

    if plot_I and not plot_V:
        min_data, max_data = np.min(I) * 1.1, np.max(I) * 1.1
    elif plot_V and not plot_I:
        min_data, max_data = np.min(V) * 1.1, np.max(V) * 1.1
    else:
        min_data, max_data = np.min(V) * 1.1, np.max(I) * 1.1

    title_start = 'RATAN-600 calibrated scan' if is_calibrated else 'RATAN-600 raw scan'
    modes = 'I-V' if plot_I and plot_V else 'I' if plot_I else 'V'
    title = f"{title_start} // {DATE} // {modes}"
    y_label = 'Spectral Flux Density, s.f.u' if is_calibrated else 'Antenna Temperature, K'

    fig, ax1 = plt.subplots(figsize=(8, 7))

    color_map = plt.get_cmap('std_gamma_2').reversed()
    norm = colors.Normalize(vmin=np.min(FREQ), vmax=np.max(FREQ))
    sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
    sm.set_array([])

    size = len(FREQ)
    
    if plot_I:
        for i in range(size):
            color = color_map(i / (size - 1))
            ax1.plot(x, I[i], color=color)
        ax1.set_ylabel('(I) ' + y_label)

    ax1.set_facecolor('black')
    ax1.set_xlim(-1.3 * SOLAR_R, 1.3 * SOLAR_R)
    ax1.set_ylim(min_data, max_data)

    if plot_V:
        ax2 = ax1.twinx() if plot_I else ax1
        for i in range(len(FREQ)):
            color = color_map(i / (size - 1))
            ax2.plot(x, V[i], color=color)
        ax2.set_ylabel('(V) ' + y_label)
        if plot_I: 
          ax1.set_ylim(bottom=-0.2)
          ax2.set_ylim(top=0.5)

    ax1.set_xlabel('Distance from solar centre, arcsec') 

    plt.plot([0, 0], [min_data, max_data], 'y--')
    plt.plot([SOLAR_R, SOLAR_R], [min_data, max_data], 'y--')
    plt.plot([-SOLAR_R, -SOLAR_R], [min_data, max_data], 'y--')
    plt.axis([-1.3 * SOLAR_R, 1.3 * SOLAR_R, min_data, max_data])
    plt.colorbar(sm, ax=ax2 if plot_V else ax1, ticks=FREQ[::10], orientation="horizontal", label='Frequency, GHz')

    plt.title(title)

    fig.tight_layout()
    plt.show()
