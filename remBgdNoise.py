import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from astropy.io import fits
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob, os
from tqdm import tqdm
from astropy import stats

# Directories
input_dir = 'input_files/'
output_dir = 'output_files/'
fig_dir = 'figures/'


def disp_noise(img, file_name, flux_max=2, control_plot=False):
    """Estimate the Gaussian Dispersion of the background counts distribution in the JWST _rate files

    Parameters
    ----------
    img : list
        The 2D table in DN/s of the rate file. 
    flux_max : float, optional
        Threshold used in the Gaussian fitting of the counts distribution. It corresponds to the limit (to the right)
        between the Gaussian profile of the background noise and the no-Gaussian wing.
    control_plot : str, optional
        Plot the counts histogram.
        
    Returns
    -------
    float
        The sigma value of the Gaussian profile.
    
    """

    def gaus(x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    xflux_flat = img.flatten()
    xflux_flat = xflux_flat[xflux_flat < flux_max]
    xflux_flat = xflux_flat[xflux_flat > -0.1]

    fig, ax = plt.subplots(figsize=(8, 6))
    n_bins = 200
    counts, bins, bars = ax.hist(xflux_flat, bins=n_bins, color='grey')

    x_data = []
    y_data = []

    for i in range(n_bins):
        x = bars[i].get_xy()[0]
        y = counts[i]
        x_data.append(x)
        y_data.append(y)

    x_data, y_data = np.array(x_data), np.array(y_data)

    popt, pcov = curve_fit(gaus, x_data, y_data)
    x_new = np.linspace(x_data[0], x_data[-1], 1000)
    fit_hist = gaus(x_new, popt[0], popt[1], popt[2])

    if control_plot:
        ax.plot(x_new, fit_hist, color='red',
                label='Gaussian Fit\nµ = {:.2e} DN/s\n σ = {:.2e} DN/s'.format(popt[1], abs(popt[2])))
        ax.axvline(x=abs(popt[2]), linestyle='--', color='black')

        ax.set_xlabel('Counts (DN/s)')
        ax.set_ylabel('# pixels')
        ax.set_xlim(-0.1, 0.3)
        ax.legend()
        fig.savefig(fig_dir + 'histBgd_' + file_name[:-5] + '.png', dpi=300)

    return popt[1] + abs(popt[2])


def corrFlux(flux_array, median):
    """Corrects a list of count values extracted from a JWST detector image by subtracting a \'median\' value.

    Parameters
    ----------
    flux_array : list
        Count values list to correct.
    median : float
        Value used to subtract the list of count values.

    Returns
    -------
    list
        List of subtracted input values.

    """

    flux_array_corr = np.copy(flux_array)

    for j in range(flux_array.shape[0]):
        flux_array_corr[j] = flux_array[j] - median

    return flux_array_corr


def corr_img_NIRSpec(img, threshold, return_means=False):
    rows, cols = img.shape
    new_img = np.zeros((rows, cols))
    means_array = []

    for i in range(cols):

        line = img[:, i]
        line_red = line[line < threshold]

        if np.count_nonzero(np.isnan(line_red)) < line_red.shape[0]:
            mean = np.nanmedian(line_red)
        else:
            mean = 0

        means_array.append(mean)

        new_line = corrFlux(line, mean)

        new_img[:, i] = new_line

    if return_means:
        return means_array, new_img
    else:
        return new_img


def corr_img_NIRCam(img, threshold, return_means=False):
    rows, cols = img.shape
    new_img = np.zeros((rows, cols))
    means_array = []

    for i in range(rows):

        line = img[i, :]
        line_red = line[line < threshold]

        if np.count_nonzero(np.isnan(line_red)) < line_red.shape[0]:
            mean = np.nanmedian(line_red)
        else:
            mean = 0

        means_array.append(mean)

        new_line = corrFlux(line, mean)

        new_img[i, :] = new_line

    if return_means:
        return means_array, new_img
    else:
        return new_img


def bgd_noise_removal(rate_files, max_count_value=0.5, threshold_factor=1.5, saveplot=False):
    """Corrects the intermediate files in the JWST pipeline (output from the Detector1 step
    with the suffix _rate) for 1/f correlated noise.

    Parameters
    ----------
    rate_files : str or list
        Names of files to be corrected. If several files are given as input, the names must be stored in a list.
    max_count_value : float, optional
        Max value used to remove values above this in the counts histogram and to adjust the profile using a Gaussian.
    threshold_factor : float, optional
        Multiplication factor applied to the width of the Gaussian (corresponding to the dispersion of the background
        values) as a threshold to exclude count values in the estimation of the median in each slit.
    saveplot : bool, optional
        If True, saves the images before and after correction, the median value map and the count histogram.
    Returns
    -------

    """

    # Instruments
    instrumes_expected = ['NIRSPEC', 'NIRCAM']
    instrumes_list = []

    print()
    print('1/f CORRELATED NOISE CORRECTION')
    print()

    if not os.path.exists(input_dir):  # The input directory do not exist
        print(
            'The \'input_files\' directory does not exist. Please create this directory and insert the files you want '
            'to correct.')

    else:

        try:
            os.makedirs(output_dir)
        except FileExistsError:  # The output directory already exists
            pass
        try:
            os.makedirs(fig_dir)
        except FileExistsError:  # The figure directory already exists
            pass

        if len(rate_files) == 0:
            print(
                'No files found. Make sure the files have the \'_rate\' suffix and the \'.fits\' extension. Also make '
                'sure the files are placed in the \'input_files\' directory.')
        else:

            print('Files to correct:')

            for n, rate_file in enumerate(rate_files):
                print(rate_file[len(input_dir):])

                instrume = fits.open(rate_file)[0].header['INSTRUME']
                instrumes_list.append(instrume)

            print()

            if len(set(instrumes_list)) == 1:  # All the files come from the same instrument

                instrume = instrumes_list[0]

                for n, rate_file in enumerate(tqdm(rate_files)):

                    nameFile = rate_file[len(input_dir):]
                    hdul = fits.open(rate_file)

                    primary_hdu = hdul[0]
                    sci_hdu = hdul[1]
                    err_hdu = hdul[2]
                    dq_hdu = hdul[3]
                    varPoisson_hdu = hdul[4]
                    varRnoise_hdu = hdul[5]
                    asdf_hdu = hdul[6]

                    imgNotCorrected = sci_hdu.data

                    # CORRECTION PART

                    # Flux dispersion estimate and background correction
                    sigNoise = disp_noise(imgNotCorrected, file_name=nameFile, flux_max=max_count_value,
                                          control_plot=saveplot)

                    if instrume == instrumes_expected[0]:  # NIRSPEC
                        meansArray, imgCorrected = corr_img_NIRSpec(imgNotCorrected,
                                                                    threshold=threshold_factor * sigNoise,
                                                                    return_means=True)
                    elif instrume == instrumes_expected[1]:  # NIRCAM
                        meansArray, imgCorrected = corr_img_NIRCam(imgNotCorrected,
                                                                   threshold=threshold_factor * sigNoise,
                                                                   return_means=True)

                    sci_hdu.data = imgCorrected

                    hdul = fits.HDUList(
                        [primary_hdu, sci_hdu, err_hdu, dq_hdu, varPoisson_hdu, varRnoise_hdu, asdf_hdu])
                    hdul.writeto(output_dir + nameFile[:-5] + '_bgdCorr' + '.fits', overwrite=True)

                    # RESULT VISUALISATION PART
                    if saveplot:

                        rows, cols = imgNotCorrected.shape
                        xAx, yAx = np.arange(cols), np.arange(rows)

                        imgStd = np.copy(imgNotCorrected)
                        imgStd[np.isnan(imgStd)] = 0
                        minNoCorr = 0.1 * stats.mad_std(imgStd)

                        # NOT CORRECTED
                        medValuesRows = []
                        medValuesCols = []

                        for i in range(rows):
                            if np.count_nonzero(np.isnan(imgNotCorrected[i, :])) < imgNotCorrected[i, :].shape[0]:
                                medValuesRows.append(np.nanmedian(imgNotCorrected[i, :]))
                            else:
                                medValuesRows.append(0)
                        for j in range(cols):
                            if np.count_nonzero(np.isnan(imgNotCorrected[:, j])) < imgNotCorrected[:, j].shape[0]:
                                medValuesCols.append(np.nanmedian(imgNotCorrected[:, j]))
                            else:
                                medValuesCols.append(0)

                        xlim = 1.1 * np.nanmax(np.abs(medValuesRows))
                        ylim = 1.1 * np.nanmax(np.abs(medValuesCols))

                        if instrume == instrumes_expected[0]:
                            vmax = 0.005
                        elif instrume == instrumes_expected[1]:
                            vmax = 2

                        fig = plt.figure(figsize=(7, 7))
                        grid = plt.GridSpec(6, 6, hspace=0.15, wspace=0.15)
                        mainAx = fig.add_subplot(grid[:-1, :-1])
                        rowsPts = fig.add_subplot(grid[:-1, -1], sharey=mainAx)
                        colsPts = fig.add_subplot(grid[-1, :-1], sharex=mainAx)

                        mainAx.imshow(imgNotCorrected, origin='lower', cmap='Greys',
                                           norm=colors.LogNorm(vmin=minNoCorr, vmax=vmax))

                        rowsPts.plot(medValuesRows, yAx, color='black', linewidth=0.8)
                        colsPts.plot(xAx, medValuesCols, color='black', linewidth=0.8)

                        rowsPts.axvline(0, color='grey', linestyle='--', zorder=-1)
                        colsPts.axhline(0, color='grey', linestyle='--', zorder=-1)

                        mainAx.xaxis.tick_top()
                        rowsPts.yaxis.tick_right()
                        rowsPts.invert_xaxis()

                        rowsPts.set_xlim(-xlim, xlim)
                        colsPts.set_ylim(-ylim, ylim)

                        fig.savefig(fig_dir + 'beforeCorr_' + nameFile[:-5] + '.png', dpi=300)

                        # CORRECTED
                        medValuesRows, medValuesCols = [], []

                        for i in range(rows):
                            if np.count_nonzero(np.isnan(imgCorrected[i, :])) < imgCorrected[i, :].shape[0]:
                                medValuesRows.append(np.nanmedian(imgCorrected[i, :]))
                            else:
                                medValuesRows.append(0)
                        for j in range(cols):
                            if np.count_nonzero(np.isnan(imgCorrected[:, j])) < imgCorrected[:, j].shape[0]:
                                medValuesCols.append(np.nanmedian(imgCorrected[:, j]))
                            else:
                                medValuesCols.append(0)

                        fig = plt.figure(figsize=(7, 7))
                        grid = plt.GridSpec(6, 6, hspace=0.15, wspace=0.15)
                        mainAx = fig.add_subplot(grid[:-1, :-1])
                        rowsPts = fig.add_subplot(grid[:-1, -1], sharey=mainAx)
                        colsPts = fig.add_subplot(grid[-1, :-1], sharex=mainAx)

                        rowsPts.invert_xaxis()

                        mainAx.imshow(imgCorrected, origin='lower', cmap='Greys',
                                           norm=colors.LogNorm(vmin=minNoCorr, vmax=vmax))

                        rowsPts.plot(medValuesRows, yAx, color='black', linewidth=0.8)
                        colsPts.plot(xAx, medValuesCols, color='black', linewidth=0.8)

                        rowsPts.axvline(0, color='grey', linestyle='--', zorder=-1)
                        colsPts.axhline(0, color='grey', linestyle='--', zorder=-1)

                        mainAx.xaxis.tick_top()
                        rowsPts.yaxis.tick_right()

                        rowsPts.set_xlim(-xlim, xlim)
                        colsPts.set_ylim(-ylim, ylim)

                        fig.savefig(fig_dir + 'afterCorr_' + nameFile[:-5] + '.png', dpi=300)

                        # MEDIAN MAP
                        meanImg = np.zeros((rows, cols))

                        if instrume == instrumes_expected[0]:
                            for i in range(cols):
                                meanLine = np.ones(cols) * meansArray[i]
                                meanImg[:, i] = meanLine

                        elif instrume == instrumes_expected[1]:
                            for i in range(rows):
                                meanLine = np.ones(rows) * meansArray[i]
                                meanImg[i, :] = meanLine

                        fig, ax = plt.subplots(figsize=(7, 7))
                        im = ax.imshow(meanImg, origin='lower', cmap='Greys', vmin=np.nanmin(meansArray),
                                       vmax=np.nanmax(meansArray))

                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        fig.colorbar(im, cax=cax, label='Counts (DN/s)')

                        fig.tight_layout()
                        fig.savefig(fig_dir + 'medianMap_' + nameFile[:-5] + '.png', dpi=300)

            else:

                print('!!! The files to correct do not all come from the same instrument. !!!')
                print('The files were obtained from the following instruments: ')

                for instrume in instrumes_list:
                    if instrume is not instrume[0]:
                        print(instrume)

    print()

    return



""" Example 

maxFlux = 0.5  # DN/s
thresholdFactor = 1.5  # Factor to apply on the dispersion value used to estimate a flux median value:
# σ-clip value = thresholdFactor * σ_dispersion

sstring = 'input_files/*rate.fits'
rateFiles = sorted(glob.glob(sstring))

bgd_noise_removal(rateFiles, max_count_value=maxFlux, threshold_factor=thresholdFactor, saveplot=True)

"""


