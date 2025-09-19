import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, name, label, folder):
    '''
    Correct photometric redshift distributions of the lens and source samples
    
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
    calibrate_folder = os.path.join(folder, 'CALIBRATE/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(calibrate_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(calibrate_folder, '{}/CORRECT/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(calibrate_folder, '{}/CORRECT/{}/'.format(tag, name)), exist_ok=True)
    
    # Truth
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/TRUTH.hdf5'.format(tag, name)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        truth_average_lens = file['lens']['average'][...]
        truth_average_source = file['source']['average'][...]
    
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
    difference_lens = truth_average_lens - average_lens
    
    correct_data_lens = numpy.maximum(data_lens + difference_lens[numpy.newaxis, :, :], 0.0)
    factor_lens = scipy.integrate.trapezoid(x=z_grid, y=correct_data_lens, axis=2)[:, :, numpy.newaxis]
    correct_data_lens = numpy.divide(correct_data_lens, factor_lens, out=numpy.zeros((data_size, bin_lens_size, z_size)), where=factor_lens > 0)
    
    correct_average_lens = numpy.mean(correct_data_lens, axis=0)
    average_factor_lens = scipy.integrate.trapezoid(x=z_grid, y=correct_average_lens, axis=1)[:, numpy.newaxis]
    correct_average_lens = numpy.divide(correct_average_lens, average_factor_lens, out=numpy.zeros((bin_lens_size, z_size)), where=average_factor_lens > 0)
    
    # Source
    data_size, bin_source_size, z_size = data_source.shape
    difference_source = truth_average_source - average_source
    
    correct_data_source = numpy.maximum(data_source + difference_source[numpy.newaxis, :, :], 0.0)
    factor_source = scipy.integrate.trapezoid(x=z_grid, y=correct_data_source, axis=2)[:, :, numpy.newaxis]
    correct_data_source = numpy.divide(correct_data_source, factor_source, out=numpy.zeros((data_size, bin_source_size, z_size)), where=factor_source > 0)
    
    correct_average_source = numpy.mean(correct_data_source, axis=0)
    average_factor_source = scipy.integrate.trapezoid(x=z_grid, y=correct_average_source, axis=1)[:, numpy.newaxis]
    correct_average_source = numpy.divide(correct_average_source, average_factor_source, out=numpy.zeros((bin_source_size, z_size)), where=average_factor_source > 0)
    
    with h5py.File(os.path.join(calibrate_folder, '{}/CORRECT/{}/{}.hdf5'.format(tag, name, label)), 'w') as file:
        file.create_group('meta')
        for key in meta.keys():
            file['meta'].create_dataset(key, data=meta[key], dtype=meta[key].dtype)
        
        file.create_group('lens')
        file['lens'].create_dataset('data', data=correct_data_lens, dtype=numpy.float32)
        file['lens'].create_dataset('average', data=correct_average_lens, dtype=numpy.float32)
        
        file.create_group('source')
        file['source'].create_dataset('data', data=correct_data_source, dtype=numpy.float32)
        file['source'].create_dataset('average', data=correct_average_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Calibrate Correct')
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