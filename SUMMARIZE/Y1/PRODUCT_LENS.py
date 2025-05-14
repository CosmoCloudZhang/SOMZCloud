import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, index, folder):
    '''
    Product summarization of the lens samples
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    numpy.random.seed(index)
    print('Index: {}'.format(index))
    
    # Path
    summarization_folder = os.path.join(folder, 'SUMMARIZE/')
    os.makedirs(os.path.join(summarization_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarization_folder, '{}/LENS/LENS{}'.format(tag, index)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # HISTOGRAM
    with h5py.File(os.path.join(summarization_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, index)), 'r') as file:
        data_lens_histogram = file['data'][...]
    
    # Model
    with h5py.File(os.path.join(summarization_folder, '{}/LENS/LENS{}/MODEL.hdf5'.format(tag, index)), 'r') as file:
        data_lens_model = file['data'][...]
    
    data_lens = numpy.sqrt(numpy.maximum(data_lens_histogram * data_lens_model, 0.0))
    data_lens = data_lens / scipy.integrate.trapezoid(x=z_grid, y=data_lens, axis=2)[:, :, numpy.newaxis]
    
    # Average
    average_lens = numpy.mean(data_lens, axis=1)
    average_lens = average_lens / scipy.integrate.trapezoid(x=z_grid, y=average_lens, axis=1)[:, numpy.newaxis]
    
    # Save
    with h5py.File(os.path.join(summarization_folder, '{}/LENS/LENS{}/PRODUCT.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('data', data=data_lens, dtype=numpy.float32)
        file.create_dataset('average', data=average_lens, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Product')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)
