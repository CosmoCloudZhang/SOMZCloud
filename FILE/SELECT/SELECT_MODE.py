import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing


def save_mode(z_lens, z_mean, z_mode, z_source, bin_lens, bin_source, mag0_lens, mag_source, width):
    """
    Save the selected samples.
    
    Parameters:
        z_lens (numpy.ndarray): The redshift grid of lens samples.
        z_mean (numpy.ndarray): The redshift mode of source samples.
        z_mode (numpy.ndarray): The redshifts of test application samples.
        z_source (numpy.ndarray): The redshift grid of source samples.
        bin_lens (numpy.ndarray): The redshift bin of lens samples.
        bin_source (numpy.ndarray): The redshift bin of source samples.
        mag0_lens (float): The magnitude threshold of lens samples.
        mag_source (numpy.ndarray): The magnitudes of test application samples. 
        width (int): The number of random samples for bootstrapping.
    
    Returns:
        tuple: The selected lens and source true redshift distributions.
    """
    # Select
    z1_lens = z_lens.min()
    z2_lens = z_lens.max()
    
    z1_source = z_source.min()
    z2_source = z_source.max()
    
    select_source = (z1_source < z_mean) & (z_mean < z2_source)
    select_lens = (z1_lens < z_mean) & (z_mean < z2_lens) & (mag_source < 4 * z_mean + 18) & (mag_source < mag0_lens)
    
    # Lens
    grid_size = z_source.size - 1
    lens_size = len(bin_lens) - 1
    lens_mode = numpy.zeros((width, lens_size, grid_size), dtype=numpy.float32)
    
    for n in range(width):
        for m in range(lens_size):
            select = select_lens & (bin_lens[m] < z_mean) & (z_mean < bin_lens[m + 1])
            z_data = numpy.random.choice(z_mode[select], z_mode[select].size, replace=True)
            lens_mode[n, m, :] = numpy.histogram(z_data, bins=z_source, range=(z1_source, z2_source), density=False)[0].astype(numpy.float32)
    
    lens_count = numpy.sum(lens_mode, axis=2)
    lens = {'data': lens_mode, 'count': lens_count}
    
    # Source
    grid_size = z_source.size - 1
    source_size = len(bin_source) - 1
    source_mode = numpy.zeros((width, source_size, grid_size), dtype=numpy.float32)
    
    for n in range(width):
        for m in range(source_size):
            select = select_source & (bin_source[m] < z_mean) & (z_mean < bin_source[m + 1])
            z_data = numpy.random.choice(z_mode[select], z_mode[select].size, replace=True)
            source_mode[n, m, :] = numpy.histogram(z_data, bins=z_source, range=(z1_source, z2_source), density=False)[0].astype(numpy.float32)
    
    source_count = numpy.sum(source_mode, axis=2)
    source = {'data': source_mode, 'count': source_count}
    
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
    estimate_name = os.path.join(data_path, 'ESTIMATE/FZB_ESTIMATE{}.hdf5'.format(index))
    
    test_data = data_store.read_file(key='test_data', path=test_name, handle_class=core.data.TableHandle)
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)
    
    # Bin
    bin_name = os.path.join(data_path, 'BIN/BIN.hdf5')
    with h5py.File(bin_name, 'r') as file:
        bin_lens = file['lens'][:].astype(numpy.float32)
        bin_source = file['source'][:].astype(numpy.float32)
    
    # Redshift
    z1_lens = 0.0
    z2_lens = 2.0
    z_lens_size = 200
    z_lens = numpy.linspace(z1_lens, z2_lens, z_lens_size + 1)
    
    z1_source = 0.0
    z2_source = 3.0
    z_source_size = 300
    z_source = numpy.linspace(z1_source, z2_source, z_source_size + 1)
    
    z_mean = numpy.concatenate(estimator().mean())
    mag_source = test_data()['photometry']['mag_i_lsst']
    z_mode = numpy.concatenate(estimator().mode(grid=z_source))
    
    # Magnitude
    width = 250
    mag0_lens = 24.1
    
    # Save Mode
    lens_mode, source_mode = save_mode(z_lens, z_mean, z_mode, z_source, bin_lens, bin_source, mag0_lens, mag_source, width)
    
    with h5py.File(os.path.join(data_path, 'LENS/LENS{}/SELECT_MODE.hdf5'.format(index)), 'w') as file:
        for key, value in lens_mode.items():
            file.create_dataset(key, data=value)
    
    with h5py.File(os.path.join(data_path, 'SOURCE/SOURCE{}/SELECT_MODE.hdf5'.format(index)), 'w') as file:
        for key, value in source_mode.items():
            file.create_dataset(key, data=value)
    
    # Return
    end = time.time()
    print('Index:{}, Time: {:.2f} minutes'.format(index, (end - start) / 60))
    return index

if __name__ == '__main__':
    
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Informer')
    PARSE.add_argument('--path', type=str, required=True, help='The path to the base folder')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the processes')
    PARSE.add_argument('--length', type=int, required=True, help='The length of the train datasets')
    
    PATH = PARSE.parse_args().path
    NUMBER = PARSE.parse_args().number
    LENGTH = PARSE.parse_args().length
    
    # Multiprocessing
    with multiprocessing.Pool(processes=NUMBER) as POOL:
        POOL.starmap(main, [(PATH, index) for index in range(1, LENGTH + 1)])