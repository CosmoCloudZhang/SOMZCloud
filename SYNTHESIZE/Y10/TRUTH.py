import os
import time
import h5py
import numpy
import scipy
import argparse
import multiprocessing


def synthesize(data, weight, z_grid, number, sample_size, random_generator):
    
    indices = numpy.arange(number + 1, dtype=numpy.int32)
    select = random_generator.choice(numpy.arange(sample_size, dtype=numpy.int32), size=number + 1, replace=True)
    
    alpha = weight[indices] / numpy.sum(weight[indices])
    beta = numpy.ravel(random_generator.dirichlet(alpha, size=1))
    
    value = numpy.maximum(numpy.sum(beta[:, numpy.newaxis, numpy.newaxis] * data[indices, :, select, :], axis=0), 0.0)
    value = value / scipy.integrate.trapezoid(x=z_grid, y=value, axis=1)[:, numpy.newaxis]
    return value


def main(tag, label, number, folder):
    '''
    Truth of the spectroscopic redshift distributions of the lens and source samples
    
    Arguments:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        number (int): The number of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Label: {}'.format(label))
    random_generator = numpy.random.default_rng(0)
    
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
    summarize_lens = numpy.zeros((number + 1, bin_lens_size, sample_size, grid_size + 1))
    summarize_source = numpy.zeros((number + 1, bin_source_size, sample_size, grid_size + 1))
    
    # Factor Lens
    xi_lens = numpy.zeros((number + 1, bin_lens_size, sample_size))
    pi_lens = numpy.zeros((number + 1, bin_lens_size, sample_size))
    gamma_lens = numpy.zeros((number + 1, bin_lens_size, sample_size))
    
    for n in range(number + 1):
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/TRUTH.hdf5'.format(tag, n)), 'r') as file:
            summarize_lens[n, :, :, :] = file['data'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/LENS/LENS{}/TRUTH.hdf5'.format(tag, n)), 'r') as file:
            xi_lens[n, :, :] = file['xi'][...]
            pi_lens[n, :, :] = file['pi'][...]
            gamma_lens[n, :, :] = file['gamma'][...]
    
    factor_xi_lens = numpy.mean(numpy.sum(numpy.square(xi_lens), axis=1), axis=1)
    factor_pi_lens = numpy.mean(numpy.sum(numpy.square(pi_lens), axis=1), axis=1)
    factor_gamma_lens = numpy.mean(numpy.sum(numpy.square(gamma_lens), axis=1), axis=1)
    
    factor_lens = factor_gamma_lens / factor_pi_lens / factor_xi_lens
    
    # Factor Source
    xi_source = numpy.zeros((number + 1, bin_source_size, sample_size))
    pi_source = numpy.zeros((number + 1, bin_source_size, sample_size))
    gamma_source = numpy.zeros((number + 1, bin_source_size, sample_size))
    
    for n in range(number + 1):
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/TRUTH.hdf5'.format(tag, n)), 'r') as file:
            summarize_source[n, :, :, :] = file['data'][...]
        
        with h5py.File(os.path.join(summarize_folder, '{}/SOURCE/SOURCE{}/TRUTH.hdf5'.format(tag, n)), 'r') as file:
            xi_source[n, :, :] = file['xi'][...]
            pi_source[n, :, :] = file['pi'][...]
            gamma_source[n, :, :] = file['gamma'][...]
    
    factor_xi_source = numpy.mean(numpy.sum(numpy.square(xi_source), axis=1), axis=1)
    factor_pi_source = numpy.mean(numpy.sum(numpy.square(pi_source), axis=1), axis=1)
    factor_gamma_source = numpy.mean(numpy.sum(numpy.square(gamma_source), axis=1), axis=1)
    
    factor_source = factor_gamma_source / factor_pi_source / factor_xi_source
    
    # Factor
    factor = {
        'ZERO': 0.0,
        'HALF': 0.5,
        'UNITY': 1.0,
        'DOUBLE': 2.0
    }
    
    # Weight
    weight_lens = numpy.power(factor_lens, factor[label])
    weight_source = numpy.power(factor_source, factor[label])
    
    # Synthesize Lens
    with multiprocessing.Pool(processes=size) as pool:
        data_lens = numpy.stack(pool.starmap(synthesize, [(summarize_lens, weight_lens, z_grid, number, sample_size, random_generator) for _ in range(synthesize_size)]), axis=0)
    
    average_lens = numpy.median(data_lens, axis=0)
    average_lens = average_lens / scipy.integrate.trapezoid(x=z_grid, y=average_lens, axis=1)[:, numpy.newaxis]
    
    # Synthesize Source
    with multiprocessing.Pool(processes=size) as pool:
        data_source = numpy.stack(pool.starmap(synthesize, [(summarize_source, weight_source, z_grid, number, sample_size, random_generator) for _ in range(synthesize_size)]), axis=0)
    
    average_source = numpy.median(data_source, axis=0)
    average_source = average_source / scipy.integrate.trapezoid(x=z_grid, y=average_source, axis=1)[:, numpy.newaxis]
    
    with h5py.File(os.path.join(synthesize_folder, '{}/TRUTH_{}.hdf5'.format(tag, label)), 'w') as file:
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
    PARSE = argparse.ArgumentParser(description='Synthesize Truth')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    LABEL = PARSE.parse_args().label
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, LABEL, NUMBER, FOLDER)
