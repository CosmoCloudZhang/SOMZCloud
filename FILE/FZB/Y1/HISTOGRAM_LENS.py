import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, index, folder):
    '''
    Histogram of the spec redshifts of the lens samples
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_bin = numpy.linspace(z1 - z_delta / 2, z2 + z_delta / 2, z_grid.size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        z_true = file['photometry']['redshift_true'][:]
    
    # Bin
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        lens_bin = file['bin'][:].astype(numpy.float32)
        lens_select = file['select'][:].astype(bool)
    lens_bin_size = len(lens_bin) - 1
    sample_size = 1000
    
    # Lens
    lens_single = numpy.zeros((lens_bin_size, grid_size + 1))
    lens_sample = numpy.zeros((lens_bin_size, sample_size, grid_size + 1))
    
    for m in range(len(lens_bin) - 1):
        # Select
        z_select = z_true[lens_select[m, :]]
        
        # Single
        histogram = numpy.histogram(z_select, bins=z_bin, range=(z1, z2), density=True)[0]
        lens_single[m, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
        
        # Sample
        for n in range(sample_size):
            z_sample = numpy.random.choice(z_select, len(z_select), replace=True)
            
            histogram = numpy.histogram(z_sample, bins=z_bin, range=(z1, z2), density=True)[0]
            lens_sample[m, n, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
    
    # Save
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('single', data=lens_single, dtype=numpy.float32)
        file.create_dataset('sample', data=lens_sample, dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Histogram')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)