import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(index, folder):
    '''
    Summarize the true redshift distribution of the lens and source samples.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The base folder of the dataset.
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    application_name = os.path.join(dataset_folder, 'APPLICATION/DATA{}.hdf5'.format(index))
    application_dataset = data_store.read_file(key='application', path=application_name, handle_class=core.data.TableHandle)()
    
    estimate_name = os.path.join(fzb_folder, 'ESTIMATE/ESTIMATE{}.hdf5'.format(index))
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    # Bin
    with h5py.File(os.path.join(fzb_folder, 'LENS/LENS{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_lens = file['bin'][:].astype(numpy.float32)
    
    with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'r') as file:
        bin_source = file['bin'][:].astype(numpy.float32)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.0
    z2_source = 3.0
    
    width = 1000
    grid_size = 300
    z_delta = (z2_source - z1_source) / grid_size
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    # Estimate
    z_mean = numpy.concatenate(estimator.mean())
    z_median = numpy.concatenate(estimator.median())
    z_mode = numpy.concatenate(estimator.mode(z_grid))
    z_true = application_dataset['photometry']['redshift']
    z_phot = numpy.average([z_mean, z_median, z_mode], axis=0)
    mag_source = application_dataset['photometry']['mag_i_lsst']
    del application_dataset, estimator, z_mean, z_median, z_mode
    
    # Select
    slope = 4.0
    intercept = 18.0
    select_source = numpy.isfinite(z_phot) & (z1_source < z_phot) & (z_phot <= z2_source)
    select_lens = numpy.isfinite(z_phot) & (z1_lens < z_phot) & (z_phot <= z2_lens) & (mag_source < slope * z_phot + intercept)
    
    # Lens
    for m in range(len(bin_lens) - 1):
        # Select
        lens_single = numpy.zeros(grid_size + 1)
        lens_sample = numpy.zeros((width, grid_size + 1))
        z_select = z_true[select_lens & (bin_lens[m] < z_phot) & (z_phot < bin_lens[m + 1])]
        
        # Single
        histogram = numpy.histogram(z_select, bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0]
        single = numpy.interp(x=z_grid, xp=(z_grid[+1:] + z_grid[:-1]) / 2, fp=histogram, left=0.0, right=0.0)
        lens_single = single / single.sum() / z_delta
        
        # Sample
        for n in range(width):
            z_sample = numpy.random.choice(z_select, z_select.size, replace=True)
            
            histogram = numpy.histogram(z_sample, bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0]
            sample = numpy.interp(x=z_grid, xp=(z_grid[+1:] + z_grid[:-1]) / 2, fp=histogram, left=0.0, right=0.0)
            lens_sample[n, :] = sample / sample.sum() / z_delta
        
        # Save
        lens_data = {'single':lens_single, 'sample': lens_sample}
        with h5py.File(os.path.join(fzb_folder, 'LENS/LENS{}/SELECT{}.hdf5'.format(index, m + 1)), 'w') as file:
            for key, value in lens_data.items():
                file.create_dataset(key, data=value)
    
    # Source
    for m in range(len(bin_source) - 1):
        # Select
        source_single = numpy.zeros(grid_size + 1)
        source_sample = numpy.zeros((width, grid_size + 1))
        z_select = z_true[select_source & (bin_source[m] < z_phot) & (z_phot < bin_source[m + 1])]
        
        # Single
        histogram = numpy.histogram(z_select, bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0]
        single = numpy.interp(x=z_grid, xp=(z_grid[+1:] + z_grid[:-1]) / 2, fp=histogram, left=0.0, right=0.0)
        source_single = single / single.sum() / z_delta
        
        # Sample
        z_sample = numpy.random.choice(z_select, z_select.size, replace=True)
        for n in range(width):
            
            histogram = numpy.histogram(z_sample, bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0]
            sample = numpy.interp(x=z_grid, xp=(z_grid[+1:] + z_grid[:-1]) / 2, fp=histogram, left=0.0, right=0.0)
            source_sample[n, :] = sample / sample.sum() / z_delta
        
        # Save
        source_data = {'single':source_single, 'sample': source_sample}
        with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/SELECT{}.hdf5'.format(index, m + 1)), 'w') as file:
            for key, value in source_data.items():
                file.create_dataset(key, data=value)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='FZB Summarize')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)