import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing

def select(z1_lens, z2_lens, z1_source, z2_source, z_mean, mag_source, mag0_lens, mag0_source):
    """
    Select the samples based on the redshift and magnitude criteria.
    
    Parameters:
        z1_lens (float): The minimum lens redshift.
        z2_lens (float): The maximum lens redshift.
        z1_source (float): The minimum source redshift.
        z2_source (float): The maximum source redshift.
        z_mean (numpy.ndarray): The mean redshift values.
        mag_source (numpy.ndarray): The source magnitudes.
        mag0_lens (float): The maximum lens magnitude.
        mag0_source (float): The maximum source magnitude.
        
    Returns:
        tuple: The selected lens and source samples.
    """
    # Select
    slope = 4.0
    intercept = 18.0
    select_source = (z1_source < z_mean) & (z_mean < z2_source) & (mag_source < mag0_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < slope * z_mean + intercept) & (mag_source < mag0_lens)
    
    return select_lens, select_source

def save_select(width, z_mean, z_true, z_edge, bin_lens, bin_source, select_lens, select_source):
    """
    Save the selected samples.
    
    Parameters:
        width (int): The number of samples to draw.
        z_mean (numpy.ndarray): The mean redshift values.
        z_true (numpy.ndarray): The true redshift values.
        z_edge (numpy.ndarray): The redshift bin edges.
        bin_lens (numpy.ndarray): The lens redshift bins.
        bin_source (numpy.ndarray): The source redshift bins.
        select_lens (numpy.ndarray): The selected lens samples.
        select_source (numpy.ndarray): The selected source samples.
    
    Returns:
        tuple: The selected lens and source true redshift distributions.
    """
    # Lens
    z_size = z_edge.size - 1
    lens_size = len(bin_lens) - 1
    lens_data = numpy.zeros((lens_size, z_size), dtype=numpy.float32)
    lens_sample = numpy.zeros((width, lens_size, z_size), dtype=numpy.float32)
    
    for n in range(width):
        for m in range(lens_size):
            select = select_lens & (bin_lens[m] <= z_mean) & (z_mean < bin_lens[m + 1])
            lens_data[m, :] = numpy.histogram(z_true[select], bins=z_edge, range=(z_edge.min(), z_edge.max()), density=False)[0].astype(numpy.float32)
            
            z_data = numpy.random.choice(z_true[select], z_true[select].size, replace=True)
            lens_sample[n, m, :] = numpy.histogram(z_data, bins=z_edge, range=(z_edge.min(), z_edge.max()), density=False)[0].astype(numpy.float32)
        
    lens_count = numpy.sum(lens_sample, axis=2)
    lens = {'data': lens_data, 'count': lens_count, 'sample': lens_sample}
    
    # Source
    z_size = z_edge.size - 1
    source_size = len(bin_source) - 1
    source_data = numpy.zeros((source_size, z_size), dtype=numpy.float32)
    source_sample = numpy.zeros((width, source_size, z_size), dtype=numpy.float32)
    
    for n in range(width):
        for m in range(source_size):
            select = select_source & (bin_source[m] <= z_mean) & (z_mean < bin_source[m + 1])
            source_data[m, :] = numpy.histogram(z_true[select], bins=z_edge, range=(z_edge.min(), z_edge.max()), density=False)[0].astype(numpy.float32)
            
            z_data = numpy.random.choice(z_true[select], z_true[select].size, replace=True)
            source_sample[n, m, :] = numpy.histogram(z_data, bins=z_edge, range=(z_edge.min(), z_edge.max()), density=False)[0].astype(numpy.float32)
            
    source_count = numpy.sum(source_sample, axis=2)
    source = {'data': source_data, 'count': source_count, 'sample': source_sample}
    
    # Return
    return lens, source


def main(path, index):
    start = time.time()
    
    # Data store
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Data
    data_path = os.path.join(path, 'DATA/')
    test_name = os.path.join(data_path, 'SAMPLE/TEST_SAMPLE.hdf5')
    estimate_name = os.path.join(data_path, 'FZB/FZB_ESTIMATE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)
    
    # Bin
    with h5py.File(os.path.join(data_path, 'SELECT/LENS/LENS{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_lens = file['bin'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(data_path, 'SELECT/SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_source = file['bin'][:].astype(numpy.float32)
    
    # Redshift
    z1_lens = 0.0
    z2_lens = 2.0
    
    z1_source = 0.0
    z2_source = 3.0
    
    z_size = 300
    z_edge = numpy.linspace(z1_source, z2_source, z_size + 1)
    
    mag0_lens = 24.1
    mag0_source = 25.5
    
    z_mean = numpy.concatenate(estimator().mean())
    z_true = test_data()['photometry']['redshift']
    mag_source = test_data()['photometry']['mag_i_lsst']
    
    # Save Select
    width = 1000
    select_lens, select_source = select(z1_lens, z2_lens, z1_source, z2_source, z_mean, mag_source, mag0_lens, mag0_source)
    lens_data, source_data = save_select(width, z_mean, z_true, z_edge, bin_lens, bin_source, select_lens, select_source)
    
    with h5py.File(os.path.join(data_path, 'SELECT/LENS/LENS{}/SELECT.hdf5'.format(index)), 'w') as file:
        for key, value in lens_data.items():
            file.create_dataset(key, data=value)
    
    with h5py.File(os.path.join(data_path, 'SELECT/SOURCE/SOURCE{}/SELECT.hdf5'.format(index)), 'w') as file:
        for key, value in source_data.items():
            file.create_dataset(key, data=value)
    
    # Save Datasets
    for k in range(len(bin_lens) - 1): 
        select_bin = select_lens & (bin_lens[k] <= z_mean) & (z_mean < bin_lens[k + 1])
        with h5py.File(os.path.join(data_path, 'SELECT/LENS/LENS{}/SELECT{}.hdf5'.format(index, k)), 'w') as file:
            for name in test_data().keys():
                file.create_group(name=name)
                for key, value in test_data()[name].items():
                    file[name].create_dataset(key, data=value[select_bin])
    
    for k in range(len(bin_source) - 1):
        select_bin = select_source & (bin_source[k] <= z_mean) & (z_mean < bin_source[k + 1])
        with h5py.File(os.path.join(data_path, 'SELECT/SOURCE/SOURCE{}/SELECT{}.hdf5'.format(index, k)), 'w') as file:
            for name in test_data().keys():
                file.create_group(name=name)
                for key, value in test_data()[name].items():
                    file[name].create_dataset(key, data=value[select_bin])
    
    # Return
    end = time.time()
    print('Index:{}, Time: {:.2f} minutes'.format(index, (end - start) / 60))
    return index

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='SELECT')
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