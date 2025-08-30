import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, label, index, folder):
    '''
    Product summarization of the source samples
    
    Arguments:
        tag (str): The tag of configuration
        label (str): The label of configuration
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
    os.makedirs(os.path.join(summarization_folder, '{}/{}/SOURCE/'.format(label, tag)), exist_ok=True)
    os.makedirs(os.path.join(summarization_folder, '{}/{}/SOURCE/SOURCE{}'.format(label, tag, index)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # DIR
    with h5py.File(os.path.join(summarization_folder, '{}/{}/SOURCE/SOURCE{}/DIR.hdf5'.format(label, tag, index)), 'r') as file:
        data_source_dir = file['data'][...]
    
    # Stack
    with h5py.File(os.path.join(summarization_folder, '{}/{}/SOURCE/SOURCE{}/STACK.hdf5'.format(label, tag, index)), 'r') as file:
        data_source_stack = file['data'][...]
    
    data_source = numpy.sqrt(numpy.maximum(data_source_dir * data_source_stack, 0.0))
    data_source = data_source / scipy.integrate.trapezoid(x=z_grid, y=data_source, axis=2)[:, :, numpy.newaxis]
    
    # Average
    average_source = numpy.mean(data_source, axis=1)
    average_source = average_source / scipy.integrate.trapezoid(x=z_grid, y=average_source, axis=1)[:, numpy.newaxis]
    
    # Save
    with h5py.File(os.path.join(summarization_folder, '{}/{}/SOURCE/SOURCE{}/PRODUCT.hdf5'.format(label, tag, index)), 'w') as file:
        file.create_dataset('data', data=data_source, dtype=numpy.float32)
        file.create_dataset('average', data=average_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Copper Product Source')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    LABEL = PARSE.parse_args().label
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, LABEL, INDEX, FOLDER)
