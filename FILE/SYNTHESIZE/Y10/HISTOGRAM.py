import os
import time
import h5py
import numpy
import scipy
import argparse
import multiprocessing


def synthesize(data, weight, z_grid, number, sample_size):
    
    indices = numpy.arange(number, dtype=numpy.int32)
    select = numpy.random.choice(numpy.arange(sample_size, dtype=numpy.int32), size=number, replace=True)
    
    alpha = weight[indices, select] / numpy.sum(weight[indices, select])
    beta = numpy.random.dirichlet(numpy.transpose(alpha), size=1).flatten()
    
    value = numpy.maximum(numpy.sum(beta[:, numpy.newaxis, numpy.newaxis] * data[indices, :, select, :], axis=0), 0.0)
    value = value / scipy.integrate.trapezoid(x=z_grid, y=value, axis=1)[:, numpy.newaxis]
    return value


def main(tag, number, folder):
    '''
    Histogram of the spectroscopic redshift distributions of the lens and source samples
    
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
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(synthesize_folder, '{}/'.format(tag)), exist_ok=True)
    
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
    size = 16
    sample_size = 100
    synthesize_size = 500000
    
    # Summarize
    summarize_lens = numpy.zeros((number, bin_lens_size, sample_size, grid_size + 1))
    summarize_source = numpy.zeros((number, bin_source_size, sample_size, grid_size + 1))
    
    # Factor Lens
    sigma_lens = numpy.zeros((number, bin_lens_size, sample_size))
    metric_lens = numpy.zeros((number, bin_lens_size, sample_size))
    fraction_lens = numpy.zeros((number, bin_lens_size, sample_size))
    
    for n in range(number):
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
            summarize_lens[n, :, :, :] = file['data'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/TARGET.hdf5'.format(tag, n + 1)), 'r') as file:
            sigma_lens[n, :, :] = file['sigma'][...]
            metric_lens[n, :, :] = file['metric'][...]
            fraction_lens[n, :, :] = file['fraction'][...]
    
    factor_sigma_lens = numpy.sum(numpy.square(sigma_lens), axis=1)
    factor_metric_lens = numpy.sum(numpy.square(metric_lens), axis=1)
    factor_fraction_lens = numpy.square(numpy.sum(fraction_lens, axis=1))
    factor_lens = factor_fraction_lens / factor_sigma_lens / factor_metric_lens
    
    # Factor Source
    sigma_source = numpy.zeros((number, bin_source_size, sample_size))
    metric_source = numpy.zeros((number, bin_source_size, sample_size))
    fraction_source = numpy.zeros((number, bin_source_size, sample_size))
    
    for n in range(number):
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/HISTOGRAM.hdf5'.format(tag, n + 1)), 'r') as file:
            summarize_source[n, :, :, :] = file['data'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/TARGET.hdf5'.format(tag, n + 1)), 'r') as file:
            sigma_source[n, :, :] = file['sigma'][...]
            metric_source[n, :, :] = file['metric'][...]
            fraction_source[n, :, :] = file['fraction'][...]
    
    factor_sigma_source = numpy.prod(sigma_source, axis=1)
    factor_metric_source = numpy.prod(metric_source, axis=1)
    factor_fraction_source = numpy.prod(fraction_source, axis=1)
    factor_source = factor_fraction_source / factor_sigma_source / factor_metric_source
    
    # Loop
    factor_list = [0.0, 0.5, 1.0, 2.0]
    label_list = ['ZERO', 'HALF', 'UNITY', 'DOUBLE']
    for factor, label in zip(factor_list, label_list):
        print('Factor: {:.1f}, Label: {}'.format(factor, label))
        
        # Weight
        weight_lens = numpy.power(factor_lens, factor)
        weight_source = numpy.power(factor_source, factor)
        
        # Synthesize Lens
        with multiprocessing.Pool(processes=size) as pool:
            data_lens = numpy.stack(pool.starmap(synthesize, [(summarize_lens, weight_lens, z_grid, number, sample_size) for _ in range(synthesize_size)]), axis=0)
        
        average_lens = numpy.median(data_lens, axis=0)
        average_lens = average_lens / scipy.integrate.trapezoid(x=z_grid, y=average_lens, axis=1)[:, numpy.newaxis]
        
        # Synthesize Source
        with multiprocessing.Pool(processes=size) as pool:
            data_source = numpy.stack(pool.starmap(synthesize, [(summarize_source, weight_source, z_grid, number, sample_size) for _ in range(synthesize_size)]), axis=0)
        
        average_source = numpy.median(data_source, axis=0)
        average_source = average_source / scipy.integrate.trapezoid(x=z_grid, y=average_source, axis=1)[:, numpy.newaxis]
        
        with h5py.File(os.path.join(synthesize_folder, '{}/HISTOGRAM_{}.hdf5'.format(tag, label)), 'w') as file:
            file.create_group('lens')
            file['lens'].create_dataset('data', data=data_lens, dtype=numpy.float32)
            file['lens'].create_dataset('average', data=average_lens, dtype=numpy.float32)
            
            file.create_group('source')
            file['source'].create_dataset('data', data=data_source, dtype=numpy.float32)
            file['source'].create_dataset('average', data=average_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Synthesize Histogram')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)
