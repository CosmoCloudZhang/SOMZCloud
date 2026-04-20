import os
import h5py
import time
import numpy
import argparse
from astropy.table import Table

def main(tag, name, label, number, folder):
    '''
    Plot the table of the assessment
    
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
    assess_folder = os.path.join(folder, 'ASSESS/')
    os.makedirs(os.path.join(assess_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/TABLE/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(assess_folder, '{}/TABLE/{}/'.format(tag, name)), exist_ok=True)
    
    # Data
    delta_mu_lens = []
    delta_mu_source = []
    
    delta_eta_lens = []
    delta_eta_source = []
    
    # Value
    for index in range(number + 1):
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/{}/{}/DATA{}.hdf5'.format(tag, name, label, index)), 'r') as file:
            mu_lens = file['lens']['average_mu'][...]
            mu_source = file['source']['average_mu'][...]
            
            eta_lens = file['lens']['average_eta'][...]
            eta_source = file['source']['average_eta'][...]
            
        with h5py.File(os.path.join(assess_folder, '{}/VALUE/{}/TRUTH/DATA{}.hdf5'.format(tag, name, index)), 'r') as file:
            truth_mu_lens = file['lens']['average_mu'][...]
            truth_mu_source = file['source']['average_mu'][...]
            
            truth_eta_lens = file['lens']['average_eta'][...]
            truth_eta_source = file['source']['average_eta'][...]
        
        delta_mu_lens.append((mu_lens - truth_mu_lens) / (1 + truth_mu_lens))
        delta_mu_source.append((mu_source - truth_mu_source) / (1 + truth_mu_source))
        
        delta_eta_lens.append((eta_lens - truth_eta_lens) / (1 + truth_mu_lens))
        delta_eta_source.append((eta_source - truth_eta_source) / (1 + truth_mu_source))
    
    average_delta_mu_lens = numpy.mean(delta_mu_lens, axis=0)
    average_delta_mu_source = numpy.mean(delta_mu_source, axis=0)
    
    average_delta_eta_lens = numpy.mean(delta_eta_lens, axis=0)
    average_delta_eta_source = numpy.mean(delta_eta_source, axis=0)
    
    sigma_delta_mu_lens = numpy.std(delta_mu_lens, axis=0)
    sigma_delta_mu_source = numpy.std(delta_mu_source, axis=0)
    
    sigma_delta_eta_lens = numpy.std(delta_eta_lens, axis=0)
    sigma_delta_eta_source = numpy.std(delta_eta_source, axis=0)
    
    # Table
    table = Table()
    table.add_column(numpy.concatenate((average_delta_mu_lens, average_delta_mu_source)), name='Average_Delta_Mu')
    table.add_column(numpy.concatenate((sigma_delta_mu_lens, sigma_delta_mu_source)), name='Sigma_Delta_Mu')
    table.add_column(numpy.concatenate((average_delta_eta_lens, average_delta_eta_source)), name='Average_Delta_Eta')
    table.add_column(numpy.concatenate((sigma_delta_eta_lens, sigma_delta_eta_source)), name='Sigma_Delta_Eta')
    
    for column in table.colnames:
        table[column].info.format = '{:+.3f}'
    
    # Save
    table.write(os.path.join(assess_folder, '{}/TABLE/{}/{}.txt'.format(tag, name, label)), format='ascii.latex', overwrite=True)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Assess Table')
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