import os
import time
import h5py
import numpy
import scipy
import argparse

import scipy.integrate
from rail import core


def main(index, folder):
    '''
    Summarize the redshift distribution of the lens samples.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The base folder of the dataset.
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(fzb_folder, 'LENS/LENS{}'.format(index)), exist_ok=True)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.0
    z2_source = 3.0
    
    grid_size = 300
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    # Bin
    with h5py.File(os.path.join(fzb_folder, 'LENS/LENS{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_lens = file['bin'][:].astype(numpy.float32)
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    combination_name = os.path.join(dataset_folder, 'COMBINATION/DATA{}.hdf5'.format(index))
    combination_dataset = data_store.read_file(key='combination', path=combination_name, handle_class=core.data.TableHandle)()
    
    mag_source = combination_dataset['photometry']['mag_i_lsst']
    z_true = combination_dataset['photometry']['redshift']
    del combination_dataset
    
    reference_name = os.path.join(fzb_folder, 'REFERENCE/REFERENCE{}.hdf5'.format(index))
    estimator = data_store.read_file(key='estimator', path=reference_name, handle_class=core.data.QPHandle)()
    
    z_pdf = estimator.pdf(z_grid)
    z_mean = numpy.concatenate(estimator.mean())
    z_median = numpy.concatenate(estimator.median())
    z_mode = numpy.concatenate(estimator.mode(z_grid))
    z_phot = numpy.average([z_mean, z_median, z_mode], axis=0)
    del z_mean, z_median, z_mode, estimator
    
    # Save Select
    slope = 4.0
    intercept = 18.0
    select_lens = numpy.isfinite(z_phot) & (z1_lens < z_phot) & (z_phot < z2_lens) & (mag_source < slope * z_phot + intercept)
    
    # Lens
    for m in range(len(bin_lens) - 1):
        select_lens_bin = select_lens & (bin_lens[m] < z_phot) & (z_phot < bin_lens[m + 1])
        
        # Summarize
        summarize_single = numpy.mean(z_pdf[select_lens_bin, :], axis=0)
        summarize_mean = scipy.integrate.trapezoid(x=z_grid, y=z_grid * summarize_single, axis=0)
        summarize_cumulative = scipy.integrate.cumulative_trapezoid(x=z_grid, y=summarize_single, axis=0, initial=0.0)
        
        # Single
        histogram = numpy.histogram(z_true[select_lens_bin], bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0]
        single = numpy.interp(x=z_grid, xp=(z_grid[+1:] + z_grid[:-1]) / 2, fp=histogram, left=0.0, right=0.0)
        
        select_single = single / scipy.integrate.trapezoid(x=z_grid, y=single, axis=0)
        select_mean = scipy.integrate.trapezoid(x=z_grid, y=z_grid * select_single, axis=0)
        select_cumulative = scipy.integrate.cumulative_trapezoid(x=z_grid, y=select_single, axis=0, initial=0.0)
        
        response = numpy.divide(select_cumulative, summarize_cumulative, out=numpy.ones_like(select_cumulative), where=summarize_cumulative > 0.0)
        with h5py.File(os.path.join(fzb_folder, 'LENS/LENS{}/RESPONSE{}.hdf5'.format(index, m + 1)), 'w') as file:
            file.create_dataset(name='response', data=response, dtype=numpy.float32)
            
            file.create_dataset(name='select_mean', data=select_mean, dtype=numpy.float32)
            file.create_dataset(name='select_single', data=select_single, dtype=numpy.float32)
            
            file.create_dataset(name='summarize_mean', data=summarize_mean, dtype=numpy.float32)
            file.create_dataset(name='summarize_single', data=summarize_single, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Response')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)