import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, name, index, folder):
    '''
    Product summarize of the lens samples
    
    Arguments:
        tag (str): The tag of configuration
        name (str): The name of configuration
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
    model_folder = os.path.join(folder, 'MODEL/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    os.makedirs(os.path.join(summarize_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/{}/LENS/'.format(tag, name)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}'.format(tag, name, index)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Target
    with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        bin_lens = file['bin_lens'][...]
    bin_lens_size = len(bin_lens) - 1
    data_size = 100
    
    # DIR
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/DIR.hdf5'.format(tag, name, index)), 'r') as file:
        data_lens_dir = file['data'][...]
    
    # Stack
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/STACK.hdf5'.format(tag, name, index)), 'r') as file:
        data_lens_stack = file['data'][...]
    
    data_lens = numpy.sqrt(numpy.maximum(data_lens_dir * data_lens_stack, 0.0))
    data_factor = scipy.integrate.trapezoid(x=z_grid, y=data_lens, axis=2)[:, :, numpy.newaxis]
    data_lens = numpy.divide(data_lens, data_factor, out=numpy.zeros((bin_lens_size, data_size, grid_size + 1)), where=data_factor > 0)
    
    # Average
    average_lens = numpy.mean(data_lens, axis=1)
    average_factor = scipy.integrate.trapezoid(x=z_grid, y=average_lens, axis=1)[:, numpy.newaxis]
    average_lens = numpy.divide(average_lens, average_factor, out=numpy.zeros((bin_lens_size, grid_size + 1)), where=average_factor > 0)
    
    # Save
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/HYBRID.hdf5'.format(tag, name, index)), 'w') as file:
        file.create_dataset('data', data=data_lens, dtype=numpy.float32)
        file.create_dataset('average', data=average_lens, dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Iron Hybrid Lens')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, INDEX, FOLDER)
