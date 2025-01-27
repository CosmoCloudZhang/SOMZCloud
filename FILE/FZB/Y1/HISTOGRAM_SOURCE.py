import os
import time
import h5py
import numpy
import scipy
import argparse
from rail import core


def main(tag, index, folder):
    '''
    Histogram of the spec redshifts of the source samples
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    sample_size = 1000
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Application
    application_name = os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index))
    application_dataset = data_store.read_file(key='application', path=application_name, handle_class=core.data.TableHandle)()
    
    z_spec = application_dataset['photometry']['redshift']
    del application_dataset
    
    # Bin
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
        source_bin = file['bin'][:].astype(numpy.float32)
        source_select = file['select'][:].astype(numpy.bool)
    
    # Lens
    for m in range(len(source_bin) - 1):
        # Select
        z_select = z_spec[source_select[m, :]]
        
        # Single
        histogram = numpy.histogram(z_select, bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0]
        single = numpy.interp(x=z_grid, xp=(z_grid[+1:] + z_grid[:-1]) / 2, fp=histogram, left=0.0, right=0.0)
        source_single = single / scipy.integrate.trapezoid(x=z_grid, y=single, axis=0)
        
        # Sample
        source_sample = numpy.zeros((sample_size, grid_size + 1))
        for n in range(sample_size):
            z_sample = numpy.random.choice(z_select, len(z_select), replace=True)
            
            histogram = numpy.histogram(z_sample, bins=z_grid, range=(z_grid.min(), z_grid.max()), density=True)[0]
            sample = numpy.interp(x=z_grid, xp=(z_grid[+1:] + z_grid[:-1]) / 2, fp=histogram, left=0.0, right=0.0)
            source_sample[n, :] = sample / scipy.integrate.trapezoid(x=z_grid, y=sample, axis=0)
        
        # Save
        source_data = {'single': source_single, 'sample': source_sample}
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
    PARSE = argparse.ArgumentParser(description='FZB Histogram')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)