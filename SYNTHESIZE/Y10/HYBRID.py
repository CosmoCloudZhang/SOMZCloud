import os
import time
import h5py
import numpy
import scipy
import argparse
import multiprocessing


def synthesize(data, z_grid, number, sample_size, random_generator):
    
    indices = numpy.arange(number, dtype=numpy.int32)
    select = random_generator.choice(numpy.arange(sample_size, dtype=numpy.int32), size=number, replace=True)
    
    alpha = numpy.ones(number) / number
    beta = numpy.ravel(random_generator.dirichlet(alpha, size=1))
    
    value = numpy.maximum(numpy.sum(beta[:, numpy.newaxis, numpy.newaxis] * data[indices, :, select, :], axis=0), 0.0)
    factor = scipy.integrate.trapezoid(x=z_grid, y=value, axis=1)[:, numpy.newaxis]
    
    return numpy.divide(value, factor, out=numpy.zeros(value.shape), where=factor != 0)


def main(tag, name, number, folder):
    '''
    Hybrid of the spectroscopic redshift distributions of the lens and source samples
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        number (int): The number of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Name: {}'.format(name))
    random_generator = numpy.random.default_rng(0)
    
    # Path
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(synthesize_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(synthesize_folder, '{}/{}/'.format(tag, name)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Bin
    with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS0/HYBRID.hdf5'.format(tag, name)), 'r') as file:
        bin_lens_size = file['meta']['bin_size'][...]
    
    with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE0/HYBRID.hdf5'.format(tag, name)), 'r') as file:
        bin_source_size = file['meta']['bin_size'][...]
    
    # Size
    size = 16
    sample_size = 100
    synthesize_size = 500000
    
    # Summarize Lens
    bin_lens = numpy.zeros((number, bin_lens_size + 1))
    summarize_lens = numpy.zeros((number, bin_lens_size, sample_size, grid_size + 1))
    
    for n in range(number):
        with h5py.File(os.path.join(summarize_folder, '{}/{}/LENS/LENS{}/HYBRID.hdf5'.format(tag, name, n + 1)), 'r') as file:
            bin_lens[n, :] = file['meta']['bin'][...]
            summarize_lens[n, :, :, :] = file['ensemble']['data'][...]
    bin_lens = numpy.mean(bin_lens, axis=0)
    
    # Summarize Source
    bin_source = numpy.zeros((number, bin_source_size + 1))
    summarize_source = numpy.zeros((number, bin_source_size, sample_size, grid_size + 1))
    
    for n in range(number):
        with h5py.File(os.path.join(summarize_folder, '{}/{}/SOURCE/SOURCE{}/HYBRID.hdf5'.format(tag, name, n + 1)), 'r') as file:
            bin_source[n, :] = file['meta']['bin'][...]
            summarize_source[n, :, :, :] = file['ensemble']['data'][...]
    bin_source = numpy.mean(bin_source, axis=0)
    
    # Synthesize Lens
    with multiprocessing.Pool(processes=size) as pool:
        data_lens = numpy.stack(pool.starmap(synthesize, [(summarize_lens, z_grid, number, sample_size, random_generator) for _ in range(synthesize_size)]), axis=0)
    
    average_lens = numpy.mean(data_lens, axis=0)
    factor_lens = scipy.integrate.trapezoid(x=z_grid, y=average_lens, axis=1)[:, numpy.newaxis]
    average_lens = numpy.divide(average_lens, factor_lens, out=numpy.zeros((bin_lens_size, grid_size + 1)), where=factor_lens != 0)
    
    # Synthesize Source
    with multiprocessing.Pool(processes=size) as pool:
        data_source = numpy.stack(pool.starmap(synthesize, [(summarize_source, z_grid, number, sample_size, random_generator) for _ in range(synthesize_size)]), axis=0)
    
    average_source = numpy.mean(data_source, axis=0)
    factor_source = scipy.integrate.trapezoid(x=z_grid, y=average_source, axis=1)[:, numpy.newaxis]
    average_source = numpy.divide(average_source, factor_source, out=numpy.zeros((bin_source_size, grid_size + 1)), where=factor_source != 0)
    
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/HYBRID.hdf5'.format(tag, name)), 'w') as file:
        file.create_group('meta')
        file['meta'].create_dataset('z1', data=z1, dtype=numpy.float32)
        file['meta'].create_dataset('z2', data=z2, dtype=numpy.float32)
        file['meta'].create_dataset('z_grid', data=z_grid, dtype=numpy.float32)
        file['meta'].create_dataset('grid_size', data=grid_size, dtype=numpy.int32)
        file['meta'].create_dataset('synthesize_size', data=synthesize_size, dtype=numpy.int32)
        
        file['meta'].create_dataset('bin_lens', data=bin_lens, dtype=numpy.float32)
        file['meta'].create_dataset('bin_lens_size', data=bin_lens_size, dtype=numpy.int32)
        
        file['meta'].create_dataset('bin_source', data=bin_source, dtype=numpy.float32)
        file['meta'].create_dataset('bin_source_size', data=bin_source_size, dtype=numpy.int32)
        
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
    PARSE = argparse.ArgumentParser(description='Synthesize Hybrid')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, NUMBER, FOLDER)