import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, index, folder):
    '''
    Histogram of the spectroscopic redshifts of the lens samples
    
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
    som_folder = os.path.join(folder, 'SOM/')
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(som_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(som_folder, '{}/LENS/LENS{}'.format(tag, index)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_delta = (z2 - z1) / grid_size
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    z_bin = numpy.linspace(z1 - z_delta / 2, z2 + z_delta / 2, z_grid.size + 1)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_cell_id = file['meta']['cell_id'][...]
        application_redshift_true = file['photometry']['redshift_true'][...]
    
    # Select
    with h5py.File(os.path.join(fzb_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_lens = file['bin_lens'][...]
    
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_lens = file['select'][...]
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_cell_id = file['meta']['cell_id'][...]
    
    # Reference
    with h5py.File(os.path.join(fzb_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_lens = file['bin_lens'][...]
    
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/REFERENCE.hdf5'.format(tag, index)), 'r') as file:
        reference_lens = file['reference'][...]
    
    # Size
    sample_size = 100
    bin_lens_size = len(bin_lens) - 1
    
    # Lens
    single_lens = numpy.zeros((bin_lens_size, grid_size + 1))
    sample_lens = numpy.zeros((bin_lens_size, sample_size, grid_size + 1))
    
    # Loop
    for m in range(bin_lens_size):
        # Select
        select = select_lens[m, :]
        select_size = numpy.sum(select)
        
        # Reference
        reference = reference_lens[m, :]
        reference_size = numpy.sum(reference)
        
        # Application
        application_cell_id_select = application_cell_id[select]
        application_redshift_true_select = application_redshift_true[select]
        
        # Combination
        combination_cell_id_reference = combination_cell_id[reference]
        
        # Weight
        weight = numpy.array(numpy.isin(application_cell_id_select, combination_cell_id_reference), dtype=numpy.float32)
        
        # Single
        histogram = numpy.histogram(application_redshift_true_select, bins=z_bin, range=(z1, z2), weights=weight, density=True)[0]
        single_lens[m, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
        
        # Bootstrap
        for n in range(sample_size):
            # Application
            application_indices = numpy.random.choice(numpy.arange(select_size), select_size, replace=True)
            application_redshift_true_sample = application_redshift_true_select[application_indices]
            application_cell_id_sample = application_cell_id_select[application_indices]
            
            # Combination
            combination_indices = numpy.random.choice(numpy.arange(reference_size), reference_size, replace=True)
            combination_cell_id_sample = combination_cell_id_reference[combination_indices]
            
            # Weight
            weight_sample = numpy.array(numpy.isin(application_cell_id_sample, combination_cell_id_sample), dtype=numpy.float32)
            
            # Sample
            histogram = numpy.histogram(application_redshift_true_sample, bins=z_bin, range=(z1, z2), weights=weight_sample, density=True)[0]
            sample_lens[m, n, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
    
    # Save
    with h5py.File(os.path.join(som_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('single', data=single_lens, dtype=numpy.float32)
        file.create_dataset('sample', data=sample_lens, dtype=numpy.float32)
    
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