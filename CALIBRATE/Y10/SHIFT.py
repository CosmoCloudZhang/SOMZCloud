import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, name, label, folder):
    '''
    Fiducial photometric redshift distributions of the lens and source samples
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Synthesize store
    start = time.time()
    print('Name: {}, Label: {}'.format(name, label))
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    calibrate_folder = os.path.join(folder, 'CALIBRATE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(calibrate_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(calibrate_folder, '{}/SHIFT/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(calibrate_folder, '{}/SHIFT/{}/'.format(tag, name)), exist_ok=True)
    
    # Value
    with h5py.File(os.path.join(analyze_folder, '{}/VALUE/{}/TRUTH.hdf5'.format(tag, name)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        truth_average_mu_lens = file['lens']['average_mu'][...]
        truth_average_mu_source = file['source']['average_mu'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/VALUE/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        mu_lens = file['lens']['mu'][...]
        mu_source = file['source']['mu'][...]
        
        average_mu_lens = file['lens']['average_mu'][...]
        average_mu_source = file['source']['average_mu'][...]
    
    # Data
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        data_lens = file['lens']['data'][...]
        data_source = file['source']['data'][...]
        
        average_lens = file['lens']['average'][...]
        average_source = file['source']['average'][...]
    
    # Redshift
    z_grid = meta['z_grid']
    
    # Lens
    data_size, bin_lens_size, z_size = data_lens.shape
    shift_lens = numpy.zeros((data_size, bin_lens_size, z_size))
    
    difference_mu_lens = truth_average_mu_lens - average_mu_lens
    zeta_lens = numpy.random.multivariate_normal(mean=difference_mu_lens, cov=numpy.cov(mu_lens, rowvar=False), size=data_size)
    
    for m in range(bin_lens_size):
        z_shift = z_grid[numpy.newaxis, :] - zeta_lens[:, m, numpy.newaxis]
        shift_lens[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, average_lens[m, :], extrapolate=True)(z_shift), 0)
    
    factor_lens = scipy.integrate.trapezoid(x=z_grid, y=shift_lens, axis=2)[:, :, numpy.newaxis]
    shift_lens = numpy.divide(shift_lens, factor_lens, out=numpy.zeros((data_size, bin_lens_size, z_size)), where=factor_lens > 0)
    
    # Source
    data_size, bin_source_size, z_size = data_source.shape
    shift_source = numpy.zeros((data_size, bin_source_size, z_size))
    
    difference_mu_source = truth_average_mu_source - average_mu_source
    zeta_source = numpy.random.multivariate_normal(mean=difference_mu_source, cov=numpy.cov(mu_source, rowvar=False), size=data_size)
    
    for m in range(bin_source_size):
        z_shift = z_grid[numpy.newaxis, :] - zeta_source[:, m, numpy.newaxis]
        shift_source[:, m, :] = numpy.maximum(scipy.interpolate.CubicSpline(z_grid, average_source[m, :], extrapolate=True)(z_shift), 0)
    
    factor_source = scipy.integrate.trapezoid(x=z_grid, y=shift_source, axis=2)[:, :, numpy.newaxis]
    shift_source = numpy.divide(shift_source, factor_source, out=numpy.zeros((data_size, bin_source_size, z_size)), where=factor_source > 0)
    
    # Save
    with h5py.File(os.path.join(calibrate_folder, '{}/SHIFT/{}/{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_group('meta')
        for key in meta.keys():
            file['meta'].create_dataset(key, data=meta[key], dtype=meta[key].dtype)
        
        file.create_group('lens')
        file['lens'].create_dataset('shift', data=shift_lens)
        
        file.create_group('source')
        file['source'].create_dataset('shift', data=shift_source)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Calibrate Shift')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, LABEL, FOLDER)