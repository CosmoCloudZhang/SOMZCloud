import os
import time
import h5py
import numpy
import scipy
import argparse
import multiprocessing


def ensemble(data, z_grid, number, sample_size):
    
    n = numpy.arange(number, dtype=numpy.int32)
    m = numpy.random.choice(numpy.arange(sample_size, dtype=numpy.int32), size=number, replace=True)
    
    alpha = numpy.random.dirichlet(numpy.ones(number), size=1).flatten()
    beta = numpy.random.dirichlet(alpha, size=1).flatten()
    
    value = numpy.maximum(numpy.sum(beta[:, numpy.newaxis, numpy.newaxis] * data[n, :, m, :], axis=0), 0.0)
    value = value / scipy.integrate.trapezoid(y=value, x=z_grid, axis=1)[:, numpy.newaxis]
    return value


def main(tag, number, folder):
    '''
    Histogram of the spectroscopic redshifts of the source samples
    
    Arguments:
        tag (str): The tag of the configuration
        number (int): The number of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    
    os.makedirs(ensemble_folder, exist_ok=True)
    os.makedirs(os.path.join(ensemble_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Data
    bin_size = 5
    sample_size = 100
    data = numpy.zeros((number, bin_size, sample_size, grid_size + 1))
    
    for n in range(number):
        
        with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
            data[n, :, :, :] = file['sample'][...]
    
    # Ensemble Data
    count = 16
    ensemble_size = 10000
    with multiprocessing.Pool(processes=count) as pool:
        ensemble_data = numpy.stack(pool.starmap(ensemble, [(data, z_grid, number, sample_size) for _ in range(ensemble_size)]), axis=0)
    
    ensemble_average = numpy.mean(ensemble_data, axis=0)
    ensemble_average = ensemble_average / scipy.integrate.trapezoid(y=ensemble_average, x=z_grid, axis=1)[:, numpy.newaxis]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/SOURCE/HISTOGRAM.hdf5'.format(tag)), 'w') as file:
        file.create_dataset('ensemble', data=ensemble_data, dtype=numpy.float32)
        file.create_dataset('average', data=ensemble_average, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Ensemble Histogram')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)
