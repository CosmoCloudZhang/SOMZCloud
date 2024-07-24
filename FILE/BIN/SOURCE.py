import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing


def select(z_mean, z_lens, z_source, mag_source):
    """
    Select the samples based on the redshift and magnitude criteria.
    
    Parameters:
        z_mean (numpy.ndarray): The mean redshift values of the samples.
        z_lens (list): The redshift range of the lens samples.
        z_source (list): The redshift range of the source samples.
        mag_source (numpy.ndarray): The magnitude values of the source samples.
        
    Returns:
        tuple: The selected lens and source samples.
    """
    # Redshift
    z1_lens, z2_lens = z_lens
    z1_source, z2_source = z_source
    
    # Select
    slope = 4.0
    intercept = 18.0
    select_source = (z1_source < z_mean) & (z_mean < z2_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < slope * z_mean + intercept)
    
    return select_lens, select_source


def bin(z, bin_size):
    """
    Calculate the redshift bins for a given range of redshifts and bin size.
    
    Parameters:
        z (numpy.ndarray): The redshift values to bin.
        bin_size (int): The number of bins to divide the redshift range into.
    
    Returns:
        numpy.ndarray: An array of redshift values representing the bin edges.
    
    """
    quantiles = numpy.linspace(0, 1, bin_size + 1)
    z_bin = numpy.quantile(z, quantiles)
    return z_bin


def main(path, index):
    """
    Select the source samples based on the redshift and magnitude criteria.
    
    Parameters:
        path (str): The path to the base folder.
        index (int): The index of the source sample.
    
    Returns:
        numpy.ndarray: The redshift bins of the selected source samples.
    
    """
    
    # Data store
    start = time.time()
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    estimate_name = os.path.join(data_path, 'FZB/FZB_ESTIMATE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    z_lens = [z1_lens, z2_lens]
    
    z1_source = 0.0
    z2_source = 3.0
    z_source = [z1_source, z2_source]
    
    bin_size = 5
    z_mean = numpy.concatenate(estimator().mean())
    mag_source = test_data()['photometry']['mag_i_lsst']
    
    # Select
    select_source = select(z_mean, z_lens, z_source, mag_source)[1]
    bin_source = bin(z_mean[select_source], bin_size)
    
    # Save
    os.makedirs(os.path.join(data_path, 'BIN/SOURCE/SOURCE{}'.format(index)), exist_ok=True)
    with h5py.File(os.path.join(data_path, 'BIN/SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'w') as file:
        file.create_dataset('bin', data=bin_source)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return bin_source
    
if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='Source Binning')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    # Multiprocessing
    SIZE = LENGTH // NUMBER
    for CHUNK in range(SIZE):
        print('CHUNK: {}'.format(CHUNK + 1))
        with multiprocessing.Pool(processes=NUMBER) as POOL:
            POOL.starmap(main, [(PATH, INDEX) for INDEX in range(CHUNK * NUMBER + 1, (CHUNK + 1) * NUMBER + 1)])