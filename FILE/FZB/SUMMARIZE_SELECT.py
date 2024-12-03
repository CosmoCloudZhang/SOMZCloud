import os
import time
import h5py
import numpy
import argparse
from rail import core
import multiprocessing


def summarize(index, folder):
    '''
    Summarize the redshift distribution of the lens and source samples.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The base folder of the dataset.
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    application_name = os.path.join(dataset_folder, 'APPLICATION/DATA{}.hdf5'.format(index + 1))
    application_dataset = data_store.read_file(key='application', path=application_name, handle_class=core.data.TableHandle)()
    
    estimate_name = os.path.join(fzb_folder, 'ESTIMATE/ESTIMATE{}.hdf5'.format(index + 1))
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    # Bin
    with h5py.File(os.path.join(fzb_folder, 'LENS/BIN.hdf5'), 'r') as file:
        bin_lens = file['bin'][index - 1, :].astype(numpy.float32)
    
    with h5py.File(os.path.join(fzb_folder, 'SOURCE/BIN.hdf5'), 'r') as file:
        bin_source = file['bin'][index - 1, :].astype(numpy.float32)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.0
    z2_source = 3.0
    
    width = 1000
    grid_size = 300
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    z_mean = numpy.concatenate(estimator.mean())
    z_true = application_dataset['photometry']['redshift']
    mag_source = application_dataset['photometry']['mag_i_lsst']
    
    # Select
    slope = 4.0
    intercept = 18.0
    select_source = numpy.isfinite(z_mean) & (z1_source < z_mean) & (z_mean <= z2_source)
    select_lens = numpy.isfinite(z_mean) & (z1_lens < z_mean) & (z_mean <= z2_lens) & (mag_source < slope * z_mean + intercept)
    
    # Lens
    lens_size = len(bin_lens) - 1
    lens_data = numpy.zeros((lens_size, grid_size), dtype=numpy.float32)
    lens_sample = numpy.zeros((width, lens_size, grid_size), dtype=numpy.float32)
    
    for n in range(width):
        for m in range(lens_size):
            select = select_lens & (bin_lens[m] <= z_mean) & (z_mean < bin_lens[m + 1])
            lens_data[m, :] = numpy.histogram(z_true[select], bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0].astype(numpy.float32)
            
            z_data = numpy.random.choice(z_true[select], z_true[select].size, replace=True)
            lens_sample[n, m, :] = numpy.histogram(z_data, bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0].astype(numpy.float32)
    lens_data = {'data': lens_data, 'sample': lens_sample}
    
    with h5py.File(os.path.join(fzb_folder, 'LENS/LENS{}/SUMMARIZE_SELECT.hdf5'.format(index)), 'w') as file:
        for key, value in lens_data.items():
            file.create_dataset(key, data=value)
    
    # Source
    source_size = len(bin_source) - 1
    source_data = numpy.zeros((source_size, grid_size), dtype=numpy.float32)
    source_sample = numpy.zeros((width, source_size, grid_size), dtype=numpy.float32)
    
    for m in range(source_size):
        select = select_source & (bin_source[m] <= z_mean) & (z_mean < bin_source[m + 1])
        source_data[m, :] = numpy.histogram(z_true[select], bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0].astype(numpy.float32)
        for n in range(width):
            z_data = numpy.random.choice(z_true[select], z_true[select].size, replace=True)
            source_sample[n, m, :] = numpy.histogram(z_data, bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0].astype(numpy.float32)
    source_data = {'data': source_data, 'sample': source_sample}
    
    with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/SUMMARIZE_SELECT.hdf5'.format(index)), 'w') as file:
        for key, value in source_data.items():
            file.create_dataset(key, data=value)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Index:{}, Time: {:.2f} minutes'.format(index, duration))
    return duration


def main(count, number, folder):
    '''
    Summarize the redshift distribution of the lens and source samples.
    
    Arguments:
        count (int): The count of the process.
        number (int): The number of the dataset.
        folder (str): The base folder of the dataset.
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Start
    start = time.time()
    
    # Multiprocessing
    size = number // count
    for chunk in range(size):
        print('CHUNK: {}'.format(chunk + 1))
        with multiprocessing.Pool(processes=count) as pool:
            pool.starmap(summarize, [(index, folder) for index in range(chunk * count, (chunk + 1) * count)])
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Duration: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarize')
    PARSE.add_argument('--count', type=int, required=True, help='The count of the process')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the dataset')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    COUNT = PARSE.parse_args().count
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(COUNT, NUMBER, FOLDER)