import os
import time
import h5py
import numpy
import scipy
import argparse


def main(tag, number, folder):
    '''
    Histogram of the spectroscopic redshifts of the lens samples
    
    Arguments:
        tag (str): The tag of the configuration
        number (int): The number of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # ensemble store
    start = time.time()
    numpy.random.seed(0)
    
    # Path
    ensemble_folder = os.path.join(folder, 'ENSEMBLE/')
    
    os.makedirs(ensemble_folder, exist_ok=True)
    os.makedirs(os.path.join(ensemble_folder, '{}/'.format(tag)), exist_ok=True)
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    grid_size = 300
    z_grid = numpy.linspace(z1, z2, grid_size + 1)
    
    # Histogram
    with h5py.File(os.path.join(ensemble_folder, '{}/HISTOGRAM.hdf5'.format(tag)), 'r') as file:
        histogram_average_lens = file['lens']['average'][...]
        histogram_average_source = file['source']['average'][...]
    
    # Product
    with h5py.File(os.path.join(ensemble_folder, '{}/PRODUCT.hdf5'.format(tag)), 'r') as file:
        product_ensemble_lens = file['lens']['ensemble'][...]
        product_ensemble_source = file['source']['ensemble'][...]
        
        product_average_lens = file['lens']['average'][...]
        product_average_source = file['source']['average'][...]
    
    # Response
    epsilon = 1e-6
    response_lens = numpy.maximum(histogram_average_lens, epsilon) / numpy.maximum(product_average_lens, epsilon)
    response_source = numpy.maximum(histogram_average_source, epsilon) / numpy.maximum(product_average_source, epsilon)
    
    # Fiducial
    fiducial_ensemble_lens = response_lens[numpy.newaxis, :, :] * product_ensemble_lens
    fiducial_ensemble_source = response_source[numpy.newaxis, :, :] * product_ensemble_source
    
    fiducial_average_lens = numpy.sum(fiducial_ensemble_lens, axis=0) / number
    fiducial_average_source = numpy.sum(fiducial_ensemble_source, axis=0) / number
    
    fiducial_average_lens = fiducial_average_lens / scipy.integrate.trapezoid(x=z_grid, y=fiducial_average_lens, axis=1)[:, numpy.newaxis]
    fiducial_average_source = fiducial_average_source / scipy.integrate.trapezoid(x=z_grid, y=fiducial_average_source, axis=1)[:, numpy.newaxis]
    
    with h5py.File(os.path.join(ensemble_folder, '{}/FIDUCIAL.hdf5'.format(tag)), 'w') as file:
        file.create_group('lens')
        file['lens'].create_dataset('average', data=fiducial_average_lens, dtype=numpy.float32)
        file['lens'].create_dataset('ensemble', data=fiducial_ensemble_lens, dtype=numpy.float32)
        
        file.create_group('source')
        file['source'].create_dataset('average', data=fiducial_average_source, dtype=numpy.float32)
        file['source'].create_dataset('ensemble', data=fiducial_ensemble_source, dtype=numpy.float32)
    
    # Return
    end = time.time()
    duration = (end - start) / 60
    
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Ensemble Fiducial')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)
