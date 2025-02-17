import os
import time
import h5py
import numpy
import scipy
import argparse

def main(tag, index, folder):
    '''
    SOM summarization of the lens samples
    
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
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    
    os.makedirs(os.path.join(summarize_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}'.format(tag, index)), exist_ok=True)
    
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
        application_sigma = file['morphology']['sigma'][...]
        application_cell_size = file['meta']['cell_size'][...]
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_cell_id = file['meta']['cell_id'][...]
        combination_cell_size = file['meta']['cell_size'][...]
        combination_redshift = file['photometry']['redshift'][...]
    combination_size = len(combination_cell_id)
    
    # Select
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_source = file['bin_source'][...]
    
    with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_source = file['select'][...]
    
    # Size
    data_size = 100
    bin_source_size = len(bin_source) - 1
    
    # Source
    data_source = numpy.zeros((bin_source_size, data_size, grid_size + 1))
    
    # Loop
    for m in range(bin_source_size):
        # Select
        select = select_source[m, :] 
        select_size = numpy.sum(select)
        
        # Application
        application_sigma_select = application_sigma[select]
        application_cell_id_select = application_cell_id[select]
        
        # Bootstrap
        for n in range(data_size):
            # Application
            application_indices = numpy.random.choice(numpy.arange(select_size), size=select_size, replace=True)
            
            application_sigma_sample = application_sigma_select[application_indices]
            application_cell_id_sample = application_cell_id_select[application_indices]
            application_cell_count_sample = numpy.bincount(application_cell_id_sample, minlength=application_cell_size, weights=1 / numpy.square(application_sigma_sample))
            
            # Combination
            combination_indices = numpy.random.choice(numpy.arange(combination_size), size=combination_size, replace=True)
            
            combination_cell_id_sample = combination_cell_id[combination_indices]
            combination_cell_count_sample = numpy.bincount(combination_cell_id_sample, minlength=combination_cell_size)
            
            combination_cell_weight_sample = numpy.divide(application_cell_count_sample, combination_cell_count_sample, out=numpy.zeros(combination_cell_size), where=combination_cell_count_sample != 0)
            combination_weight_sample = combination_cell_weight_sample[combination_cell_id_sample]
            
            # Sample
            histogram = numpy.histogram(combination_redshift[combination_indices], bins=z_bin, weights=combination_weight_sample)[0]
            data_source[m, n, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
    
    # Save
    with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/SOM.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('data', data=data_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize SOM')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)
