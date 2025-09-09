import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, label, folder):
    '''
    Fiducial photometric redshift distributions of the lens and source samples
    
    Arguments:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Synthesize store
    start = time.time()
    print('Label: {}'.format(label))
    
    # Path
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(synthesize_folder, '{}/'.format(tag)), exist_ok=True)
    
    # Product
    with h5py.File(os.path.join(synthesize_folder, '{}/PRODUCT_{}.hdf5'.format(tag, label)), 'r') as file:
        product_data_lens = file['lens']['data'][...]
        product_data_source = file['source']['data'][...]
        
        product_average_lens = file['lens']['average'][...]
        product_average_source = file['source']['average'][...]
    
    # Truth
    with h5py.File(os.path.join(synthesize_folder, '{}/TRUTH_{}.hdf5'.format(tag, label)), 'r') as file:
        truth_average_lens = file['lens']['average'][...]
        truth_average_source = file['source']['average'][...]
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Fiducial
    product_response_lens = truth_average_lens - product_average_lens
    product_response_source = truth_average_source - product_average_source
    
    fiducial_data_lens = numpy.maximum(product_data_lens + product_response_lens[numpy.newaxis, :, :], 0.0)
    fiducial_data_source = numpy.maximum(product_data_source + product_response_source[numpy.newaxis, :, :], 0.0)
    
    fiducial_data_lens = fiducial_data_lens / scipy.integrate.trapezoid(x=z_grid, y=fiducial_data_lens, axis=2)[:, :, numpy.newaxis]
    fiducial_data_source = fiducial_data_source / scipy.integrate.trapezoid(x=z_grid, y=fiducial_data_source, axis=2)[:, :, numpy.newaxis]
    
    fiducial_average_lens = numpy.median(fiducial_data_lens, axis=0)
    fiducial_average_source = numpy.median(fiducial_data_source, axis=0)
    
    fiducial_average_lens = fiducial_average_lens / scipy.integrate.trapezoid(x=z_grid, y=fiducial_average_lens, axis=1)[:, numpy.newaxis]
    fiducial_average_source = fiducial_average_source / scipy.integrate.trapezoid(x=z_grid, y=fiducial_average_source, axis=1)[:, numpy.newaxis]
    
    with h5py.File(os.path.join(synthesize_folder, '{}/FIDUCIAL_{}.hdf5'.format(tag, label)), 'w') as file:
        file.create_group('lens')
        file['lens'].create_dataset('data', data=fiducial_data_lens, dtype=numpy.float32)
        file['lens'].create_dataset('average', data=fiducial_average_lens, dtype=numpy.float32)
        
        file.create_group('source')
        file['source'].create_dataset('data', data=fiducial_data_source, dtype=numpy.float32)
        file['source'].create_dataset('average', data=fiducial_average_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Synthesize Fiducial')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, LABEL, FOLDER)