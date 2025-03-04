import os
import time
import h5py
import numpy
import scipy
import argparse
import multiprocessing


def ensemble(data, weight, z_grid, number, sample_size):
    
    n = numpy.arange(number, dtype=numpy.int32)
    m = numpy.random.choice(numpy.arange(sample_size, dtype=numpy.int32), size=number, replace=True)
    
    alpha = weight[n, m] / numpy.sum(weight[n, m])
    beta = numpy.random.dirichlet(numpy.transpose(alpha), size=1).flatten()
    
    value = numpy.maximum(numpy.sum(beta[:, numpy.newaxis, numpy.newaxis] * data[n, :, m, :], axis=0), 0.0)
    value = value / scipy.integrate.trapezoid(x=z_grid, y=value, axis=1)[:, numpy.newaxis]
    return value


def main(tag, number, folder):
    '''
    Histogram of the spectroscopic redshifts of the lens samples
    
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
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    
    os.makedirs(ensemble_folder, exist_ok=True)
    os.makedirs(os.path.join(ensemble_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Bin
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA0.hdf5'.format(tag)), 'r') as file:
        bin_lens_size = len(file['bin_lens'][...]) - 1
    
    # Data
    sample_size = 100
    metric = numpy.zeros((number))
    fraction_lens = numpy.zeros((number, bin_lens_size, sample_size))
    data_lens = numpy.zeros((number, bin_lens_size, sample_size, grid_size + 1))
    
    for n in range(number):
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/SOM.hdf5'.format(tag, n + 1)), 'r') as file:
            data_lens[n, :, :, :] = file['data'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
            fraction_lens[n, :, :] = file['fraction'][...]
        
        with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, n + 1)), 'r') as file:
            metric[n] = file['meta']['metric'][...]
    
    weight_lens = numpy.square(scipy.stats.mstats.gmean(fraction_lens, axis=1) / metric[:, numpy.newaxis])
    
    # Ensemble Data
    count = 32
    ensemble_size = 500000
    with multiprocessing.Pool(processes=count) as pool:
        ensemble_data = numpy.stack(pool.starmap(ensemble, [(data_lens, weight_lens, z_grid, number, sample_size) for _ in range(ensemble_size)]), axis=0)
    
    ensemble_average = numpy.mean(ensemble_data, axis=0)
    ensemble_average = ensemble_average / scipy.integrate.trapezoid(x=z_grid, y=ensemble_average, axis=1)[:, numpy.newaxis]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/LENS/SOM.hdf5'.format(tag)), 'w') as file:
        file.create_dataset('data', data=ensemble_data, dtype=numpy.float32)
        file.create_dataset('average', data=ensemble_average, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Ensemble SOM')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)
