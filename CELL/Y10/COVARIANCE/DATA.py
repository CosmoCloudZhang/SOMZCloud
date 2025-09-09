import os
import h5py
import json
import time
import numpy
import argparse
from astropy import table


def main(tag, label, folder):
    '''
    Calculate the shape-shape angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        label (str): The label of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    cell_folder = os.path.join(folder, 'CELL/')
    info_folder = os.path.join(folder, 'INFO/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    
    os.makedirs(os.path.join(cell_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/COVARIANCE/'.format(tag)), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(synthesize_folder, '{}/TRUTH_{}.hdf5'.format(tag, label)), 'r') as file:
        truth_average_lens = file['lens']['average'][...]
        truth_average_source = file['source']['average'][...]
    
    # Size
    bin_lens_size, z_size = truth_average_lens.shape
    bin_source_size, z_size = truth_average_source.shape
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    z_grid = numpy.linspace(z1, z2, z_size)
    
    table_lens = table.Table()
    table_lens['redshift'] = z_grid
    for m in range(bin_lens_size):
        table_lens['n_{}(z)'.format(m + 1)] = truth_average_lens[m, :]
    table_lens.write(os.path.join(cell_folder, '{}/COVARIANCE/LENS_{}.ascii'.format(tag, label)), overwrite = True, format = 'ascii')
    
    table_source = table.Table()
    table_source['redshift'] = z_grid
    for m in range(bin_source_size):
        table_source['n_{}(z)'.format(m + 1)] = truth_average_source[m, :]
    table_source.write(os.path.join(cell_folder, '{}/COVARIANCE/SOURCE_{}.ascii'.format(tag, label)), overwrite = True, format = 'ascii')
    
    # Galaxy
    with open(os.path.join(info_folder, 'GALAXY.json'), 'r') as file:
        galaxy = json.load(file)
    
    table_bias = table.Table()
    table_bias['redshift'] = z_grid
    for m in range(bin_lens_size):
        table_bias['b_{}(z)'.format(m + 1)] = galaxy[tag]
    table_bias.write(os.path.join(cell_folder, '{}/COVARIANCE/BIAS.ascii'.format(tag)), overwrite = True, format = 'ascii')
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Cell Covariance')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    LABEL = PARSE.parse_args().label
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, LABEL, FOLDER)