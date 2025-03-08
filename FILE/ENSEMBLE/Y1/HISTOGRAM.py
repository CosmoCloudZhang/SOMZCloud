import os
import time
import h5py
import numpy
import scipy
import argparse
import multiprocessing


def ensemble(data, weight, z_grid, number, sample_size):
    
    indices = numpy.arange(number, dtype=numpy.int32)
    select = numpy.random.choice(numpy.arange(sample_size, dtype=numpy.int32), size=number, replace=True)
    
    alpha = weight[indices, select] / numpy.sum(weight[indices, select])
    beta = numpy.random.dirichlet(numpy.transpose(alpha), size=1).flatten()
    
    value = numpy.maximum(numpy.sum(beta[:, numpy.newaxis, numpy.newaxis] * data[indices, :, select, :], axis=0), 0.0)
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
    numpy.random.seed(0)
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    
    os.makedirs(ensemble_folder, exist_ok=True)
    os.makedirs(os.path.join(ensemble_folder, '{}/'.format(tag)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Bin
    with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA0.hdf5'.format(tag)), 'r') as file:
        bin_lens_size = len(file['bin_lens'][...]) - 1
        bin_source_size = len(file['bin_source'][...]) - 1
    
    # Size
    size = 32
    sample_size = 100
    ensemble_size = 500000
    
    # Data
    data_lens = numpy.zeros((number, bin_lens_size, sample_size, grid_size + 1))
    data_source = numpy.zeros((number, bin_source_size, sample_size, grid_size + 1))
    
    # Weight Lens
    sigma_lens = numpy.zeros((number, bin_lens_size, sample_size))
    metric_lens = numpy.zeros((number, bin_lens_size, sample_size))
    fraction_lens = numpy.zeros((number, bin_lens_size, sample_size))
    
    for n in range(number):
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
            data_lens[n, :, :, :] = file['data'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
            sigma_lens[n, :, :] = file['sigma'][...]
            metric_lens[n, :, :] = file['metric'][...]
            fraction_lens[n, :, :] = file['fraction'][...]
    
    mean_metric_lens = numpy.sqrt(numpy.sum(numpy.square(metric_lens), axis=1) / numpy.sum(fraction_lens, axis=1))
    mean_sigma_lens = numpy.sqrt(numpy.sum(numpy.square(sigma_lens), axis=1) / numpy.sum(fraction_lens, axis=1))
    weight_lens = 1 / numpy.square(mean_sigma_lens) / numpy.square(mean_metric_lens)
    
    # Weight Source
    sigma_source = numpy.zeros((number, bin_source_size, sample_size))
    metric_source = numpy.zeros((number, bin_source_size, sample_size))
    fraction_source = numpy.zeros((number, bin_source_size, sample_size))
    
    for n in range(number):
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
            data_source[n, :, :, :] = file['data'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
            sigma_source[n, :, :] = file['sigma'][...]
            metric_source[n, :, :] = file['metric'][...]
            fraction_source[n, :, :] = file['fraction'][...]
    
    mean_metric_source = numpy.sqrt(numpy.sum(numpy.square(metric_source), axis=1) / numpy.sum(fraction_source, axis=1))
    mean_sigma_source = numpy.sqrt(numpy.sum(numpy.square(sigma_source), axis=1) / numpy.sum(fraction_source, axis=1))
    weight_source = 1 / numpy.square(mean_sigma_source) / numpy.square(mean_metric_source)
    
    # Weight
    weight = weight_lens * weight_source
    
    # Ensemble Lens
    with multiprocessing.Pool(processes=size) as pool:
        ensemble_lens = numpy.stack(pool.starmap(ensemble, [(data_lens, weight, z_grid, number, sample_size) for _ in range(ensemble_size)]), axis=0)
    
    average_lens = numpy.mean(ensemble_lens, axis=0)
    average_lens = average_lens / scipy.integrate.trapezoid(x=z_grid, y=average_lens, axis=1)[:, numpy.newaxis]
    
    # Ensemble Source
    with multiprocessing.Pool(processes=size) as pool:
        ensemble_source = numpy.stack(pool.starmap(ensemble, [(data_source, weight, z_grid, number, sample_size) for _ in range(ensemble_size)]), axis=0)
    
    average_source = numpy.mean(ensemble_source, axis=0)
    average_source = average_source / scipy.integrate.trapezoid(x=z_grid, y=average_source, axis=1)[:, numpy.newaxis]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/HISTOGRAM.hdf5'.format(tag)), 'w') as file:
        file.create_group('lens')
        file['lens'].create_dataset('average', data=average_lens, dtype=numpy.float32)
        file['lens'].create_dataset('ensemble', data=ensemble_lens, dtype=numpy.float32)
        
        file.create_group('source')
        file['source'].create_dataset('average', data=average_source, dtype=numpy.float32)
        file['source'].create_dataset('ensemble', data=ensemble_source, dtype=numpy.float32)
    
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
