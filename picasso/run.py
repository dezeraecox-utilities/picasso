import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from picasso.picasso.gaussmle import locs_from_fits
from picasso.picasso.io import (TiffMap, load_locs, load_spots, load_tif,
                            save_datasets, save_locs)
from picasso.picasso.localize import (fit, fit_async, get_spots,
                                  identifications_from_futures, identify,
                                  localize, locs_from_fits)
from picasso.picasso.postprocess import undrift
from picasso.picasso.render import render
import time
from asyncio import wait
# from skimage import io


from loguru import logger
logger.info('Import OK')



def pylocalise(input_path, image_name, output_folder, drift_frames=False, default_info=False, trim=500, visualise=10, minimum_ng=25000, box=7):

    if not default_info:
        default_info = {
            'baseline': 100,
            'sensitivity': 1.0000,
            'gain': 55,
            'qe': 1,
            'pixelsize': 107,
        }
    # ---------------------------Localisation---------------------------
    # read in test image, identify spots
    movie, info = load_tif(f'{input_path}')
    movie = movie[trim:].copy()
    # Preview loaded image
    # plt.imshow(movie[0, :, :], cmap='Greys_r')
    logger.info('Starting identify')
    identifications = identify(movie, minimum_ng=minimum_ng, box=box, threaded=True)
    logger.info('Finished identify')

    # visualise identified spots
    python_spots = pd.DataFrame(identifications)
    for frame, df in python_spots.groupby('frame'):
        if frame > visualise:
            continue
        plt.imshow(movie[frame, :, :], cmap='Greys_r')
        sns.scatterplot(
            data=df,
            x='x',
            y='y',
            color="None",
            edgecolor='red'
            )
        plt.axis('off')
        plt.show()

    # Get spots
    logger.info('Getting spots')
    identifications.sort(kind="mergesort", order="frame"),
    spots = get_spots(movie, identifications,  box=7, camera_info=default_info)
    logger.info('Finished getting spots')

    save_datasets(path=f'{output_folder}{image_name}_spots.hdf5', info=info, spots=spots)

    # ------------------------------Fitting------------------------------
    # Using  multithreading 
    logger.info('Started async localising from fit')
    logger.info('Starting fit')
    current, thetas, CRLBs, likelihoods, iterations = fit_async(movie=movie, camera_info=default_info, identifications=identifications, box=7, method='sigmaxy', max_it=1000) # Returns current, thetas, CRLBs, likelihoods, iterations

    while current[0] < len(identifications):
        time.sleep(0.000001)
    logger.info('Finished fit')

    localisations = locs_from_fits(identifications, thetas, CRLBs, likelihoods, iterations, 7)
    save_locs(path=f'{output_folder}{image_name}_locs.hdf5', locs=localisations, info=info)

    logger.info('Finished async localising from fit')

    # ----------------------------DRIFT CORRECTION----------------------------
    locs, info = load_locs(path=f'{output_folder}{image_name}_locs.hdf5')
    if not drift_frames:
        drift_frames = int(movie.shape[0]/5)
        if drift_frames < 1:
            logger.info(f'Unable to correct drift for {image_name} due to insufficient frames')
            return 
    drift, undrift_locs = undrift(
        locs,
        info,
        segmentation=drift_frames,
        display=True,
        segmentation_callback=None,
        rcc_callback=None,
    )
    drift_fig = plt.gca()
    plt.savefig(f'{output_folder}{image_name}_drift.png')

    save_locs(path=f'{output_folder}{image_name}_locs_corr.hdf5', locs=undrift_locs, info=info)

    spots_df = pd.DataFrame(undrift_locs)
    spots_df['X [nm]'] = spots_df['x']
    spots_df['Y [nm]'] = spots_df['y']
    spots_df.to_csv(f'{output_folder}{image_name}_locs_corr.csv')

    pd.DataFrame(drift).to_csv(f'{output_folder}{image_name}_drift.csv')

    logger.info('Finished adjusting drift')

    logger.info(f'Analysis for {image_name} complete.')

if __name__ == "__main__":

    image_name = 'STORM_substack'
    drift_frames = 1250
    input_path = f'experimental_data/{image_name}.tif'
    output_folder = 'results/picasso_script/'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)()
   
    # test fit using default parameters from picasso
    pylocalise(input_path, image_name, output_folder, drift_frames=drift_frames)