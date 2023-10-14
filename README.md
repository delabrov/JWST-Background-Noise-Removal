# JWST-Background-Noise-Removal
Routine used to correct the 1/f correlated noise observed in the detector images of the JWST instruments.

# About The Project 
All the JWST data show a particular structure known as '1/f correlated noise' (Rauscher et al. 2011). Indeed, all the instruments on the telescope are connected to a pixel reading device (SIDECAR ASIC), which generates a 'banded' structure in the detector images background. Because of the random processes at its origin, it is difficult to characterise it. 

The routine proposed here is to be used on the output files from the first stage of the reduction (Detector1), which have the suffix '_rate'. As the structure of the files is not modified, the routine can be included inside your reduction code. It is important to note that the output files have the suffix '_bdgCorr'. You will then need to modify your association file for the second stage of the reduction (Image2 or Spec2) (.json extension). 

Noise in 1/f creates straight, "striped" structures in the detector images, characteristic of the way in which the pixel values are read. The correction is applied as follows: in each slit (row or column of pixels, depending on the instrument) a median of the pixel values is estimated, considering a "threshold" value for which the pixel values are excluded from the calculation of the median. This method corresponds to a sigma-clipping method. The threshold value is estimated from the "Gaussian" dispersion of all the pixel values in the detector image.

⚠️ The routine is used to correct NIRCam images and NIRSpec data-cubes. However, it cannot yet correct MIRI images and MRS data. A future version including this is planned. 

⚠️ Please cite this work and/or reference Delabrosse et al. 2023 (in prep.) if you use this project with your JWST observations.

# Getting Started 


## Requirements 
To use the routine, you need to have installed the 'jwst' module for reducing JWST data. If this is not the case, please refer to [<u>**this page**</u>](https://jwst-pipeline.readthedocs.io/en/latest/getting_started/quickstart.html). 

Before using the routine, it is important to create a directory named 'input_files' containing files with the suffix '_rate'.

## Installation 
To use the correction routine, the easiest way is to save the 'remBgdNoise.py' file in the directory containing your data reduction code. Otherwise, pay attention to the directory tree of your files when you want to import the functions into your main file. 

# Usage Example 
Here is an example of how to use the routine for NIRSpec IFU observations: (Obviously the code is not complete, but gives an idea of ​​how to use the routine)

```python 
import jwst, glob
from jwst.pipeline import Detector1Pipeline, Spec2Pipeline, Spec3Pipeline
from remBgdNoise import bgd_noise_removal # The routine that corrects the 1/f correlated noise

# First step: Detector 1
det1 = Detector1Pipeline()
det1.save_results = True
results = det1(uncal_files)

# Application of the correction routine
sstring = 'input_files/*rate.fits'
rateFiles = sorted(glob.glob(sstring))

bgd_noise_removal(rateFiles, saveplot=True) # The files are saved directly in an 'output_files' folder
# Default parameters can be modified (see function documentation)

# Second step: Spec2
spec2 = Spec2Pipeline()
spec2.save_results = True
result = spec2(rate_files)

# Third step: Spec3
spec3 = Spec3Pipeline()
spec3.save_results = True                
result = spec3(asn_file)

```

Beforehand, it is preferable to create the 'output files' and 'figures' directories which will respectively contain the noise-corrected files and the figures showing the images before and after correction. If this is not the case the routine will create the necessary directories.

<p align="center">
	<img src="https://github.com/delabrov/JWST-Background-Noise-Removal/blob/main/figures/beforeCorr_jw01644006001_05101_00001_nrs2_rate.png" width="400">
	<img src="https://github.com/delabrov/JWST-Background-Noise-Removal/blob/main/figures/afterCorr_jw01644006001_05101_00001_nrs2_rate.png" width="400">
</p>

The two Figures above show the impact of 1/f noise correction on a detector image from the NIRSpec instrument. The Figure on the left shows the initial image, without noise correction (output file from 'Detector1'). The Figure on the right shows this same image but after running the correction routine. The bottom and right panels of the image show the average pixel values ​​in both dimensions.

<p align="center">
	<img src="https://github.com/delabrov/JWST-Background-Noise-Removal/blob/main/figures/medianMap_jw01644006001_05101_00001_nrs2_rate.png" width="400">
</p>

The last Figure shows a map of the estimated median values ​​for each column of pixels in the image. These values ​​are the ones used to subtract the pixel values.

# References 
Rauscher, B. J., Arendt, R. G., Fixen, D. J., et al. 2011, in Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, Vol. 8155, Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, 81550C

# Credits
* [**Valentin Delabrosse**](https://github.com/delabrov) : Creator of the project

# Contacts 
If you have any problems, please contact me by e-mail: valentin.delabrosse@univ-grenoble-alpes.fr
