import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(number, folder):
    '''
    Bin the lens and source samples.
    
    Arguments:
        number (int): The number of mock datasets.
        folder (str): The base folder of the datasets.
    
    Returns:
        float: The duration of the process.
    '''
    # Start
    start = time.time()
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    
    # Data
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    # Bin
    lens_size = 5
    bin_lens = numpy.zeros((number, lens_size + 1))
    
    source_size = 5
    quantiles = numpy.linspace(0, 1, source_size + 1)
    bin_source = numpy.zeros((number, source_size + 1))
    
    for index in range(number):
        print('Index: {}'.format(index + 1))
        estimate_name = os.path.join(fzb_folder, 'ESTIMATE/ESTIMATE{}.hdf5'.format(index + 1))
        estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.TableHandle)()
        
        # Lens
        z1_lens = 0.2
        z2_lens = 1.2
        bin_lens[index, :] = numpy.linspace(z1_lens, z2_lens, lens_size + 1)
        
        z1_source = 0.0
        z2_source = 3.0
        z_mean = numpy.concatenate(estimator.mean())
        
        select_source = numpy.isfinite(z_mean) & (z1_source < z_mean) & (z_mean <= z2_source)
        bin_source[index, :] = numpy.quantile(z_mean[select_source], quantiles)
    # Save
    os.makedirs(os.path.join(fzb_folder, 'LENS'), exist_ok=True)
    with h5py.File(os.path.join(fzb_folder, 'LENS/BIN.hdf5'), 'w') as file:
        file.create_dataset('bin', data=bin_lens)
    
    os.makedirs(os.path.join(fzb_folder, 'SOURCE'), exist_ok=True)
    with h5py.File(os.path.join(fzb_folder, 'SOURCE/BIN.hdf5'), 'w') as file:
        file.create_dataset('bin', data=bin_source)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Binning')
    PARSE.add_argument('--number', type=int, help='The number of mock datasets')
    PARSE.add_argument('--folder', type=str, help='The path to the base folder')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)