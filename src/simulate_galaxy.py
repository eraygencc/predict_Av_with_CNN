import numpy as np
import galsim

def simulate_galaxy(tau, image_size=64, pixel_scale=0.234): #one can adjust the size of the image cutout or pixel scale accoridng to survey properties
    """
    Simulate a single galaxy image with dust extinction, PSF convolution, and noise.
    """
    # A light profile with intrinsic galaxy properties
    gal = galsim.Sersic(
        n=1.5, #Sersic index
        half_light_radius=0.7, #HLR in units of arcseconds
        flux=1e2) #total flux of the galaxy
    
    # Apply dust extinction
    gal = gal.withFlux(gal.flux * np.exp(-tau)) #for simplicity we apply extinction to all pixels
    
    # PSF
    psf = galsim.Gaussian(fwhm=0.7) #Gaussian PSF for convolution to account for seeing and telescope optics
    
    # Convolve galaxy with that PSF
    final = galsim.Convolve([gal, psf])
    
    # Draw image
    image = final.drawImage(
        nx=image_size,
        ny=image_size,
        scale=pixel_scale
    )
    
    # Add photometric noise arising from detector read noise and photon shot noise
    image.addNoise(galsim.GaussianNoise(sigma=0.02))
    
    return image.array
