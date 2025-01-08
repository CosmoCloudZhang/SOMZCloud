import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(index, folder):
    '''
    Bin the lens and source samples.
    
    Arguments:
        index (int): The index of the dataset.
        folder (str): The base folder of the datasets.
    
    Returns:
        float: The duration of the process.
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    os.makedirs(os.path.join(fzb_folder, 'LENS'), exist_ok=True)
    os.makedirs(os.path.join(fzb_folder, 'SOURCE'), exist_ok=True)
    
    # Bin
    lens_size = 5
    source_size = 5
    quantiles = numpy.linspace(0, 1, source_size + 1)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.0
    z2_source = 3.0
    
    grid_size = 300
    z_grid = numpy.linspace(z1_source, z2_source, grid_size + 1)
    
    # Load
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    estimate_name = os.path.join(fzb_folder, 'ESTIMATE/ESTIMATE{}.hdf5'.format(index))
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    z_mean = numpy.concatenate(estimator.mean())
    z_median = numpy.concatenate(estimator.median())
    z_mode = numpy.concatenate(estimator.mode(z_grid))
    z_phot = numpy.average([z_mean, z_median, z_mode], axis=0)
    del z_mean, z_median, z_mode, estimator
    
    bin_lens = numpy.linspace(z1_lens, z2_lens, lens_size + 1)
    select_source = numpy.isfinite(z_phot) & (z1_source < z_phot) & (z_phot < z2_source)
    
    bin_source = numpy.quantile(z_phot[select_source], quantiles)
    bin_source[-1] = z2_source
    bin_source[0] = z1_source
    
    # Save
    os.makedirs(os.path.join(fzb_folder, 'LENS/LENS{}'.format(index)), exist_ok=True)
    with h5py.File(os.path.join(fzb_folder, 'LENS/LENS{}/BIN.hdf5'.format(index)), 'w') as file:
        file.create_dataset('bin', data=bin_lens)
    
    os.makedirs(os.path.join(fzb_folder, 'SOURCE/SOURCE{}'.format(index)), exist_ok=True)
    with h5py.File(os.path.join(fzb_folder, 'SOURCE/SOURCE{}/BIN.hdf5'.format(index)), 'w') as file:
        file.create_dataset('bin', data=bin_source)
    
    # Delete
    del z_phot, bin_lens, bin_source, select_source
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Tomography Binning')
    PARSE.add_argument('--index', type=int, help='The index of the dataset')
    PARSE.add_argument('--folder', type=str, help='The path to the base folder')
    
    # Parse
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(INDEX, FOLDER)