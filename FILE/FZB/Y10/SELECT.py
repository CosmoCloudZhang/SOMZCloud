import os
import time
import h5py
import numpy
import argparse
from rail import core


def main(tag, index, folder):
    '''
    Define the lens and source selection
    
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
    os.makedirs(os.path.join(fzb_folder, '{}/LENS/LENS{}/'.format(tag, index)), exist_ok=True)
    
    os.makedirs(os.path.join(fzb_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/'.format(tag, index)), exist_ok=True)
    
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
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_label = file['meta']['label'][:].astype(numpy.int32)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_label = file['meta']['label'][:].astype(numpy.int32)
        application_redshift = file['photometry']['redshift'][:].astype(numpy.float32)
        application_magnitude = file['photometry']['mag_i_lsst'][:].astype(numpy.float32)
    application_size = len(application_magnitude)
    
    # Estimate
    estimate_name = os.path.join(fzb_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index))
    estimator = data_store.read_file(key='estimator', path=estimate_name, handle_class=core.data.QPHandle)()
    
    z_pdf = estimator.pdf(z_grid)
    z_mean = numpy.concatenate(estimator.mean())
    z_median = numpy.concatenate(estimator.median())
    z_mode = numpy.concatenate(estimator.mode(z_grid))
    z_phot = numpy.average([z_mean, z_median, z_mode], axis=0)
    del z_mean, z_median, z_mode, estimator
    
    # Select
    slope = 4.0
    intercept = 18.0
    
    select = numpy.isin(application_label, numpy.unique(combination_label))
    select_source = select & numpy.isfinite(z_phot) & (z1_source < z_phot) & (z_phot < z2_source)
    select_lens = select & numpy.isfinite(z_phot) & (z1_lens < z_phot) & (z_phot < z2_lens) & (application_magnitude < slope * z_phot + intercept)
    
    # Bin
    lens_size = 10
    bin_lens = numpy.linspace(z1_lens, z2_lens, lens_size + 1)
    
    source_size = 5
    quantiles = numpy.linspace(0, 1, source_size + 1)
    bin_source = numpy.quantile(z_phot[select_source], quantiles)
    bin_source[-1] = z2_source
    bin_source[0] = z1_source
    
    # Lens
    z_pdf_lens = []
    z_phot_lens = []
    z_spec_lens = []
    select_lens_bin = numpy.ones((lens_size, application_size), dtype=bool)
    
    for m in range(len(bin_lens) - 1):
        select_lens_bin[m, :] = select_lens & (bin_lens[m] < z_phot) & (z_phot < bin_lens[m + 1])
        z_spec_lens.append(application_redshift[select_lens_bin[m, :]])
        z_phot_lens.append(z_phot[select_lens_bin[m, :]])
        z_pdf_lens.append(z_pdf[select_lens_bin[m, :], :])
    
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'w') as file:    
        file.create_dataset('bin', data=bin_lens)
        file.create_dataset('select', data=select_lens_bin)
        file.create_dataset('z_pdf', data=numpy.vstack(z_pdf_lens))
        file.create_dataset('z_phot', data=numpy.concatenate(z_phot_lens, axis=0))
        file.create_dataset('z_spec', data=numpy.concatenate(z_spec_lens, axis=0))
    
    # Source
    z_pdf_source = []
    z_phot_source = []
    z_spec_source = []
    select_source_bin = numpy.ones((source_size, application_size), dtype=bool)
    
    for m in range(len(bin_source) - 1):
        select_source_bin[m, :] = select_source & (bin_source[m] < z_phot) & (z_phot < bin_source[m + 1])
        z_spec_source.append(application_redshift[select_source_bin[m, :]])
        z_phot_source.append(z_phot[select_source_bin[m, :]])
        z_pdf_source.append(z_pdf[select_source_bin[m, :], :])
    
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('bin', data=bin_source)
        file.create_dataset('select', data=select_source_bin)
        file.create_dataset('z_pdf', data=numpy.vstack(z_pdf_source))
        file.create_dataset('z_phot', data=numpy.concatenate(z_phot_source, axis=0))
        file.create_dataset('z_spec', data=numpy.concatenate(z_spec_source, axis=0))
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Selection')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)