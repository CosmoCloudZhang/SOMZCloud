import os
import h5py
import json
import time
import numpy
import argparse
from astropy import table


def main(tag, name, folder):
    '''
    Calculate information for covariance matrix of angular power spectra
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        folder (str): The base folder of the dataset
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Name: {}'.format(name))
    
    # Path
    cell_folder = os.path.join(folder, 'CELL/')
    info_folder = os.path.join(folder, 'INFO/')
    synthesize_folder = os.path.join(folder, 'SYNTHESIZE/')
    os.makedirs(os.path.join(cell_folder, '{}/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/COVARIANCE/'.format(tag)), exist_ok = True)
    os.makedirs(os.path.join(cell_folder, '{}/COVARIANCE/{}'.format(tag, name)), exist_ok = True)
    
    # Load
    with h5py.File(os.path.join(synthesize_folder, '{}/{}/TRUTH.hdf5'.format(tag, name)), 'r') as file:
        meta = {key: file['meta'][key][...] for key in file['meta'].keys()}
        
        truth_average_lens = file['lens']['average'][...]
        truth_average_source = file['source']['average'][...]
    
    # Meta
    z_grid = meta['z_grid']
    bin_lens_size = meta['lens_bin_size'][...]
    bin_source_size = meta['source_bin_size'][...]
    
    # Galaxy
    with open(os.path.join(info_folder, 'GALAXY.json'), 'r') as file:
        galaxy = json.load(file)
    
    table_bias = table.Table()
    table_bias['redshift'] = z_grid
    for m in range(bin_lens_size):
        table_bias['b_{}(z)'.format(m + 1)] = galaxy[tag]
    table_bias.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/BIAS.ascii'.format(tag, name)), overwrite = True, format = 'ascii')
    
    # Lens
    table_lens = table.Table()
    table_lens['redshift'] = z_grid
    for m in range(bin_lens_size):
        table_lens['n_{}(z)'.format(m + 1)] = truth_average_lens[m, :]
    table_lens.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/LENS.ascii'.format(tag, name)), overwrite = True, format = 'ascii')
    
    # Source
    table_source = table.Table()
    table_source['redshift'] = z_grid
    for m in range(bin_source_size):
        table_source['n_{}(z)'.format(m + 1)] = truth_average_source[m, :]
    table_source.write(os.path.join(cell_folder, '{}/COVARIANCE/{}/SOURCE.ascii'.format(tag, name)), overwrite = True, format = 'ascii')
    
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
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the dataset')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, FOLDER)