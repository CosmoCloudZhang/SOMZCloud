import os
import h5py
import time
import numpy
import argparse
from astropy.table import Table

def main(tag, name, label, number, folder):
    '''
    Plot the table of the analyze
    
    Arguments:
        tag (str): The tag of the configuration
        name (str): The name of the configuration
        label (str): The label of the configuration
        number (int): The number of the configurations
        folder (str): The base folder of the configuration
    
    Returns:
        duration (float): The duration of the process
    '''
    start = time.time()
    print('Name: {}, Label: {}'.format(name, label))
    
    # Path
    analyze_folder = os.path.join(folder, 'ANALYZE/')
    os.makedirs(os.path.join(analyze_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/TABLE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(analyze_folder, '{}/TABLE/{}/'.format(tag, name)), exist_ok=True)
    
    # Data
    zeta_lens = []
    zeta_source = []
    
    gamma_lens = []
    gamma_source = []
    
    # Value
    with h5py.File(os.path.join(analyze_folder, '{}/VALUE/{}/{}.hdf5'.format(tag, name, label)), 'r') as file:
        mu_lens = file['lens']['mu'][...]
        mu_source = file['source']['mu'][...]
        
        eta_lens = file['lens']['eta'][...]
        eta_source = file['source']['eta'][...]
    
    with h5py.File(os.path.join(analyze_folder, '{}/VALUE/{}/TRUTH.hdf5'.format(tag, name)), 'r') as file:
        truth_mu_lens = file['lens']['average_mu'][...]
        truth_mu_source = file['source']['average_mu'][...]
        
        truth_eta_lens = file['lens']['average_eta'][...]
        truth_eta_source = file['source']['average_eta'][...]
    
    zeta_lens = truth_mu_lens - mu_lens
    zeta_source = truth_mu_source - mu_source
    
    gamma_lens = truth_eta_lens / eta_lens
    gamma_source = truth_eta_source / eta_source
    
    average_zeta_lens = numpy.mean(zeta_lens, axis=0)
    average_zeta_source = numpy.mean(zeta_source, axis=0)
    
    average_gamma_lens = numpy.mean(gamma_lens, axis=0)
    average_gamma_source = numpy.mean(gamma_source, axis=0)
    
    sigma_zeta_lens = numpy.std(zeta_lens, axis=0)
    sigma_zeta_source = numpy.std(zeta_source, axis=0)
    
    sigma_gamma_lens = numpy.std(gamma_lens, axis=0)
    sigma_gamma_source = numpy.std(gamma_source, axis=0)
    
    # Table
    table = Table()
    table.add_column(numpy.concatenate((average_zeta_lens, average_zeta_source)), name='Average_Zeta')
    table.add_column(numpy.concatenate((sigma_zeta_lens, sigma_zeta_source)), name='Sigma_Zeta')
    table.add_column(numpy.concatenate((average_gamma_lens, average_gamma_source)), name='Average_Gamma')
    table.add_column(numpy.concatenate((sigma_gamma_lens, sigma_gamma_source)), name='Sigma_Gamma')
    
    for column in table.colnames:
        table[column].info.format = '{:+.3f}'
    
    # Save
    table.write(os.path.join(analyze_folder, '{}/TABLE/{}/{}.txt'.format(tag, name, label)), format='ascii.latex', overwrite=True)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Analyze Table')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of the configuration')
    PARSE.add_argument('--label', type=str, required=True, help='The label of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the configurations')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the configuration')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    LABEL = PARSE.parse_args().label
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, LABEL, NUMBER, FOLDER)