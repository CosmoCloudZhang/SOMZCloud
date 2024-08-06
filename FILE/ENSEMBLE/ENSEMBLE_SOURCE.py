import os
import time
import h5py
import numpy
import argparse


def main(path, size, width, length):
    """
    This function computes the ensemble average of the SOURCEing data.
    
    Arguments:    
        path : str : the path to the base folder
        size : int : the size of the tomography bins
        width : int : the size of the bootstrap samples
        length : int : the length of the train datasets
    
    Returns:
        duration : float : the time taken to compute the ensemble average
    """
    # Data
    start = time.time()
    data_path = os.path.join(path, 'DATA/')
    
    os.makedirs(os.path.join(data_path, 'ENSEMBLE/'), exist_ok=True)
    os.makedirs(os.path.join(data_path, 'ENSEMBLE/SOURCE/'), exist_ok=True)
    
    # Redshift
    grid_size = 300
    height = length * width
    ensemble_sample = numpy.zeros((height, size, grid_size), dtype=numpy.float32)
    
    # Ensemble
    for n in range(length):
        data_name = os.path.join(data_path, 'BIN/SOURCE/SOURCE{}/SELECT.hdf5'.format(n + 1))
        with h5py.File(data_name, 'r') as file:
            ensemble_sample[n * width: (n + 1) * width, :, :] = file['sample'][:].astype(numpy.float32)
    ensemble_data = numpy.mean(ensemble_sample, axis=0)
    
    # Save
    with h5py.File(os.path.join(data_path, 'ENSEMBLE/SOURCE/ENSEMBLE.hdf5'), 'w') as file:
        file.create_dataset('data', data=ensemble_data, dtype=numpy.float32)
        file.create_dataset('sample', data=ensemble_sample, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='Ensemble')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--size', type=int, required=True, help='The size of the tomography bins')
    PARSE.add_argument('--width', type=int, required=True, help='The size of the bootstrap samples')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    SIZE = PARSE.parse_args().size
    WIDTH = PARSE.parse_args().width
    LENGTH = PARSE.parse_args().length
    main(PATH, SIZE, WIDTH, LENGTH)