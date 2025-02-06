import os
import time
import h5py
import numpy
import scipy
import argparse

import scipy.integrate
import scipy.interpolate


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
    os.makedirs(os.path.join(fzb_folder, '{}/SELECT/'.format(tag)), exist_ok=True)
    
    os.makedirs(os.path.join(fzb_folder, '{}/LENS/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(fzb_folder, '{}/LENS/LENS{}/'.format(tag, index)), exist_ok=True)
    
    os.makedirs(os.path.join(fzb_folder, '{}/SOURCE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/'.format(tag, index)), exist_ok=True)
    
    # Redshift
    z1_lens = 0.2
    z2_lens = 1.2
    
    z1_source = 0.1
    z2_source = 2.9
    
    z1 = 0.0
    z2 = 3.0
    
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    factor = 10
    z_delta = (z2 - z1) / grid_size / factor
    z_mesh = numpy.linspace(z1, z2, grid_size * factor + 1)
    
    # Combination
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_label = file['meta']['label'][:].astype(numpy.int32)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_label = file['meta']['label'][:].astype(numpy.int32)
        application_redshift = file['photometry']['redshift'][:].astype(numpy.float32)
        application_magnitude = file['photometry']['mag_i_lsst'][:].astype(numpy.float32)
        application_redshift_true = file['photometry']['redshift_true'][:].astype(numpy.float32)
    application_size = len(application_magnitude)
    
    # Estimate
    z_mode = numpy.zeros(application_size, dtype=numpy.float32)
    z_mean = numpy.zeros(application_size, dtype=numpy.float32)
    
    chunk_size = 100000
    estimator = h5py.File(os.path.join(fzb_folder, '{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, index)), 'r')
    
    for m in range(application_size // chunk_size + 1):
        begin = m * chunk_size
        stop = min((m + 1) * chunk_size, application_size)
        
        z_pdf = scipy.interpolate.CubicSpline(x=z_grid, y=estimator['data']['yvals'][begin: stop].astype(numpy.float32), axis=1, bc_type='natural', extrapolate='False')(z_mesh)
        z_pdf = z_pdf / numpy.sum(z_pdf, axis=1, keepdims=True) / z_delta
        
        z_mode[begin: stop] = z_mesh[numpy.argmax(z_pdf, axis=1)]
        z_mean[begin: stop] = numpy.average(numpy.vstack([z_mesh] * (stop - begin)), weights=z_pdf, axis=1)
    z_phot = numpy.sqrt(z_mode * z_mean, where=z_mode * z_mean > 0, out=numpy.zeros(application_size, dtype=numpy.float32))
    
    # Select
    slope = 4.0
    intercept = 18.0
    select = numpy.isin(application_label, numpy.unique(combination_label))
    
    select_source = select & (z1_source <= z_phot) & (z_phot < z2_source)
    select_lens = select & (z1_lens <= z_phot) & (z_phot < z2_lens) & (application_magnitude < slope * z_phot + intercept)
    
    # Save
    with h5py.File(os.path.join(fzb_folder, '{}/SELECT/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('z_phot', data=z_phot)
        file.create_dataset('z_mode', data=z_mode)
        file.create_dataset('z_mean', data=z_mean)
    
    # Bin
    lens_size = 5
    bin_lens = numpy.linspace(z1_lens, z2_lens, lens_size + 1)
    
    source_size = 5
    quantiles = numpy.linspace(0, 1, source_size + 1)
    
    bin_source = numpy.quantile(z_phot[select_source], quantiles)
    bin_source[-1] = z2_source
    bin_source[0] = z1_source
    
    # Lens
    select_lens_bin = numpy.ones((lens_size, application_size), dtype=bool)
    for m in range(len(bin_lens) - 1):
        select_lens_bin[m, :] = select_lens & (bin_lens[m] <= z_phot) & (z_phot < bin_lens[m + 1])
    
    z_phot_lens = z_phot[select_lens]
    z_spec_lens = application_redshift[select_lens]
    z_true_lens = application_redshift_true[select_lens]
    
    sigma_lens = numpy.abs(z_phot_lens - z_true_lens) / (1 + z_true_lens)
    metric_lens = 1.4826 * numpy.median(numpy.abs(sigma_lens - numpy.median(sigma_lens)))
    
    fraction_lens = len(sigma_lens[sigma_lens > 0.15]) / len(sigma_lens) * 100
    percentile_lens = len(sigma_lens[numpy.abs(z_phot_lens - z_true_lens) > 1.0]) / len(sigma_lens) * 100
    
    with h5py.File(os.path.join(fzb_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'w') as file:    
        file.create_dataset('bin', data=bin_lens)
        file.create_dataset('select', data=select_lens_bin)
        
        file.create_dataset('z_phot', data=z_phot_lens)
        file.create_dataset('z_spec', data=z_spec_lens)
        file.create_dataset('z_true', data=z_true_lens)
        
        file.create_dataset('metric', data=metric_lens)
        file.create_dataset('fraction', data=fraction_lens)
        file.create_dataset('percentile', data=percentile_lens)
    
    # Source
    select_source_bin = numpy.ones((source_size, application_size), dtype=bool)
    for m in range(len(bin_source) - 1):
        select_source_bin[m, :] = select_source & (bin_source[m] <= z_phot) & (z_phot < bin_source[m + 1])
    
    z_phot_source = z_phot[select_source]
    z_spec_source = application_redshift[select_source]
    z_true_source = application_redshift_true[select_source]
    
    sigma_source = numpy.abs(z_phot_source - z_true_source) / (1 + z_true_source)
    metric_source = 1.4826 * numpy.median(numpy.abs(sigma_source - numpy.median(sigma_source)))
    
    fraction_source = len(sigma_source[sigma_source > 0.15]) / len(sigma_source) * 100
    percentile_source = len(sigma_source[numpy.abs(z_phot_source - z_true_source) > 1.0]) / len(sigma_source) * 100
    
    with h5py.File(os.path.join(fzb_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'w') as file:
        file.create_dataset('bin', data=bin_source)
        file.create_dataset('select', data=select_source_bin)
        
        file.create_dataset('z_phot', data=z_phot_source)
        file.create_dataset('z_spec', data=z_spec_source)
        file.create_dataset('z_true', data=z_true_source)
        
        file.create_dataset('metric', data=metric_source)
        file.create_dataset('fraction', data=fraction_source)
        file.create_dataset('percentile', data=percentile_source)
    
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