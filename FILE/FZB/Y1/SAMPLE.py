import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(tag, index, folder):
    '''
    Define the lens and source samples
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        float: The duration of the process
    '''
    # Start
    start = time.time()
    print('Index:{}'.format(index))
    
    # Path
    fzb_folder = os.path.join(folder, 'FZB/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(fzb_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(fzb_folder, '{}/LENS/{}/'.format(tag, index)), exist_ok=True)
    
    os.makedirs(os.path.join(fzb_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(fzb_folder, '{}/SOURCE/{}/'.format(tag, index)), exist_ok=True)
    
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
    
    # Application
    application_name = os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index))
    application_dataset = data_store.read_file(key='application', path=application_name, handle_class=core.data.TableHandle)()
    
    magnitude = application_dataset['photometry']['mag_i_lsst']
    del application_dataset
    
    # Estimate
    estimate_name = os.path.join(fzb_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index))
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    z_mean = numpy.concatenate(estimator.mean())
    z_median = numpy.concatenate(estimator.median())
    z_mode = numpy.concatenate(estimator.mode(z_grid))
    z_phot = numpy.average([z_mean, z_median, z_mode], axis=0)
    del z_mean, z_median, z_mode, estimator
    
    # Select
    slope = 4.0
    intercept = 18.0
    indices = numpy.arange(len(z_phot))
    select_source = numpy.isfinite(z_phot) & (z1_source < z_phot) & (z_phot < z2_source)
    select_lens = numpy.isfinite(z_phot) & (z1_lens < z_phot) & (z_phot < z2_lens) & (magnitude < slope * z_phot + intercept)
    
    # Bin
    lens_size = 5
    bin_lens = numpy.linspace(z1_lens, z2_lens, lens_size + 1)
    
    source_size = 5
    quantiles = numpy.linspace(0, 1, source_size + 1)
    bin_source = numpy.quantile(z_phot[select_source], quantiles)
    bin_source[-1] = z2_source
    bin_source[0] = z1_source
    
    # Lens
    select_lens_bin = numpy.ones((lens_size, len(z_phot)), dtype=bool)
    for m in range(len(bin_lens) - 1):
        select_lens_bin[m, :] = select_lens & (bin_lens[m] < z_phot) & (z_phot < bin_lens[m + 1])
    
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SAMPLE.hdf5'.format(tag, index)), 'w') as file:    
        file.create_dataset('bin', data=bin_lens)
        file.create_dataset('sample', data=indices[select_lens_bin])
    
    # Source
    select_source_bin = numpy.ones((source_size, len(z_phot)), dtype=bool)
    for m in range(len(bin_source) - 1):
        select_source_bin[m, :] = select_source & (bin_source[m] < z_phot) & (z_phot < bin_source[m + 1])
        
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/SAMPLE.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('bin', data=bin_source)
        file.create_dataset('sample', data=indices[select_source_bin])
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Sample Definition')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)