import os
import h5py
import json
import time
import numpy
import argparse


def main(number, folder):
    '''
    Store the fiducial values of density configuration
    
    Arguments:
        number (int): The number of the datasets
        folder (str): The base folder of the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    info_folder = os.path.join(folder, 'INFO/')
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Definition
    slope = 4.0
    delta = 0.05
    intercept = 18.0
    magnification = {'Y1': {}, 'Y10': {}}
    
    # Loop
    for tag in magnification.keys():
        print('Tag: {}'.format(tag))
        
        value = []
        for n in range(number):
            print('Index: {}'.format(n + 1))
            
            # Application
            with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, n + 1)), 'r') as file:
                application_magnitude = file['photometry']['mag_i_lsst'][...]
            
            # Target
            with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, n + 1)), 'r') as file:
                z_phot = file['z_phot'][...]
                bin_lens = file['bin_lens'][...]
            bin_lens_size = len(bin_lens) - 1
            
            count_plus = numpy.zeros(bin_lens_size, dtype=int)
            target_lens_plus = (application_magnitude < slope * z_phot + intercept + delta)
            for m in range(bin_lens_size):
                count_plus[m] = numpy.sum(target_lens_plus & (bin_lens[m] <= z_phot) & (z_phot < bin_lens[m + 1]))
            
            count_minus = numpy.zeros(bin_lens_size, dtype=int)
            target_lens_minus = (application_magnitude < slope * z_phot + intercept - delta)
            for m in range(bin_lens_size):
                count_minus[m] = numpy.sum(target_lens_minus & (bin_lens[m] <= z_phot) & (z_phot < bin_lens[m + 1]))
            
            value.append(numpy.log10(count_plus / count_minus) / 2 / delta)
        magnification[tag] = numpy.mean(numpy.vstack(value), axis=0).tolist()
    
    # Save
    with open(os.path.join(info_folder, 'MAGNIFICATION.json'), 'w') as file:
        json.dump(magnification, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Magnification')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    # Output
    OUTPUT = main(NUMBER, FOLDER)