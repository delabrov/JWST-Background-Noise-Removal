# JWST-Background-Noise-Removal
Routine used to correct the 1/f correlated noise observed in the detector images of the JWST instruments.


## Table of contents 
- Description
- Requirements
- Installation
- Example
- References
- Contacts


## Decription 
All the JWST data show a particular structure known as '1/f correlated noise' (Rauscher et al. 2011). Indeed, all the instruments on the telescope are connected to a pixel reading device (SIDECAR ASIC), which generates a 'banded' structure in the detector images background. Because of the random processes at its origin, it is difficult to characterise it. 

The routine proposed here is to be used on the output files from the first stage of the reduction (Detector1), which have the suffix '_rate'. As the structure of the files is not modified, the routine can be included inside your reduction code. It is important to note that the output files have the suffix '_bdgCorr'. You will then need to modify your association file for the second stage of the reduction (Image2 or Spec2) (.json extension). 

Noise in 1/f creates straight, "striped" structures in the detector images, characteristic of the way in which the pixel values are read. The correction is applied as follows: in each slit (row or column of pixels, depending on the instrument) a median of the pixel values is estimated, considering a "threshold" value for which the pixel values are excluded from the calculation of the median. This method corresponds to a sigma-clipping method. The threshold value is estimated from the "Gaussian" dispersion of all the pixel values in the detector image. 

## Requirements 
To use the routine, you need to have installed the 'jwst' module for reducing JWST data. If this is not the case, please refer to: https://jwst-pipeline.readthedocs.io/en/latest/getting_started/quickstart.html

## Installation 
To use the correction routine, the easiest way is to save the 'remBgdNoise.py' file in the directory containing your data reduction code. Otherwise, pay attention to the directory tree of your files when you want to import the functions into your main file. 

## Example 

## References 
Rauscher, B. J., Arendt, R. G., Fixen, D. J., et al. 2011, in Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, Vol. 8155, Society of Photo-Optical Instrumentation Engineers (SPIE) Conference Series, 81550C

## Contacts 
If you have any problems, please contact me by e-mail: valentin.delabrosse@univ-grenoble-alpes.fr
