import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, index, folder):
    '''
    Histogram of the spectroscopic redshifts of the source samples
    
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
    
    os.makedirs(os.path.join(fzb_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}'.format(tag, index)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_bin = numpy.linspace(z1 - z_delta / 2, z2 + z_delta / 2, z_grid.size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_sigma = file['morphology']['sigma'][...]
        application_redshift_true = file['photometry']['redshift_true'][...]
    
    # Select
    with h5py.File(os.path.join(fzb_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
    
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_source = file['select'][...]
    
    # Size
    sample_size = 100
    bin_source_size = len(bin_source) - 1
    
    # Source
    single_source = numpy.zeros((bin_source_size, grid_size + 1))
    sample_source = numpy.zeros((bin_source_size, sample_size, grid_size + 1))
    
    for m in range(bin_source_size):
        # Select
        select = select_source[m, :]
        select_size = numpy.sum(select)
        
        # Application
        application_redshift_true_select = application_redshift_true[select]
        application_sigma_select = application_sigma[select]
        
        # Weight
        weight = 1 / numpy.square(application_sigma_select)
        
        # Single
        histogram = numpy.histogram(application_redshift_true_select, bins=z_bin, range=(z1, z2), weights=weight, density=True)[0]
        single_source[m, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
        
        # Sample
        for n in range(sample_size):
            
            # Application
            application_indices = numpy.random.choice(numpy.arange(select_size), select_size, replace=True)
            application_redshift_true_sample = application_redshift_true_select[application_indices]
            application_sigma_sample = application_sigma_select[application_indices]
            
            # Weight
            weight_sample = 1 / numpy.square(application_sigma_sample)
            
            histogram = numpy.histogram(application_redshift_true_sample, bins=z_bin, range=(z1, z2), weights=weight_sample, density=True)[0]
            sample_source[m, n, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
    
    # Save
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('single', data=single_source, dtype=numpy.float32)
        file.create_dataset('sample', data=sample_source, dtype=numpy.float32)
    
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