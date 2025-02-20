import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, index, folder):
    '''
    Model summarization of the lens samples
    
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
    summarization_folder = os.path.join(folder, 'SUMMARIZE/')
    
    os.makedirs(os.path.join(summarization_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarization_folder, '{}/LENS/LENS{}'.format(tag, index)), exist_ok=True)
    
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
        application_cell_size = file['meta']['cell_size'][...]
        application_size = len(file['photometry']['redshift'][...])
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_cell_id = file['meta']['cell_id'][...]
        combination_cell_size = file['meta']['cell_size'][...]
        combination_redshift = file['photometry']['redshift'][...]
    combination_size = len(combination_cell_id)
    
    # Select
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_lens = file['bin_lens'][...]
    
    with h5py.File(os.path.join(model_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        select_lens = file['select'][...]
    
    # Size
    data_size = 100
    bin_lens_size = len(bin_lens) - 1
    data_lens = numpy.zeros((bin_lens_size, data_size, grid_size + 1))
    
    # Estimator
    estimator = h5py.File(os.path.join(model_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index)), 'r')
    
    # Loop
    for m in range(bin_lens_size):
        # Select
        select = select_lens[m, :] 
        select_size = numpy.sum(select)
        
        # PDF
        select_indices = numpy.arange(application_size)[select]
        z_pdf = estimator['data']['yvals'][select_indices].astype(numpy.float32)
        
        # Application
        application_cell_id_select = application_cell_id[select]
        
        # Loop
        for n in range(data_size):
            # Application
            application_indices = numpy.random.choice(numpy.arange(select_size), size=select_size, replace=True)
            
            application_cell_id_data = application_cell_id_select[application_indices]
            application_cell_count_data = numpy.bincount(application_cell_id_data, minlength=application_cell_size)
            
            # Combination
            combination_indices = numpy.random.choice(numpy.arange(combination_size), size=combination_size, replace=True)
            
            combination_z_spec_data = combination_redshift[combination_indices]
            combination_cell_id_data = combination_cell_id[combination_indices]
            combination_cell_count_data = numpy.bincount(combination_cell_id_data, minlength=combination_cell_size)
            
            combination_cell_weight_data = numpy.divide(application_cell_count_data, combination_cell_count_data, out=numpy.zeros(combination_cell_size), where=combination_cell_count_data != 0)
            combination_weight_data = combination_cell_weight_data[combination_cell_id_data]
            if n == 0:
                print(combination_weight_data.min(), combination_weight_data.max())
            histogram = numpy.zeros((grid_size + 1))
            for k in range(combination_cell_size):
                filter = combination_cell_id_data == k
                histogram_som = numpy.histogram(combination_z_spec_data[filter], bins=z_bin, weights=combination_weight_data[filter], range=(z1, z2), density=False)[0]
                
                filter = application_cell_id_data == k
                histogram_model = numpy.sum(z_pdf[application_indices, :][filter, :], axis=0)
                
                histogram = histogram + numpy.sqrt(histogram_som * histogram_model)
            data_lens[m, n, :] = histogram / scipy.integrate.trapezoid(x=z_grid, y=histogram, axis=0)
    
    # Save
    with h5py.File(os.path.join(summarization_folder, '{}/LENS/LENS{}/FIDUCIAL.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('data', data=data_lens, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Fiducial')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)
