import os
import yaml
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
    
    # Load
    with open(os.path.join(dataset_folder, 'CATALOG/OBSERVE.yaml'), 'r') as file:
        observe = yaml.safe_load(file)
        observation_list = observe['healpix_pixels']
        area = observe['sky_area'] / len(observation_list) * len(observation_list) // 2
    
    # Definition
    sigma0 = 0.26
    density = {'Y1': {}, 'Y10': {}}
    
    # Loop
    for tag in density.keys():
        print('Tag: {}'.format(tag))
        
        # Density
        density_lens = []
        density_source = []
        
        for n in range(number):
            print('Index: {}'.format(n + 1))
            
            # Target
            with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, n + 1)), 'r') as file:
                bin_lens_size = len(file['bin_lens'][...]) - 1
                bin_source_size = len(file['bin_source'][...]) - 1
            
            # Application
            with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, n + 1)), 'r') as file:
                application_sigma = file['morphology']['sigma'][...]
            
            # Lens
            with h5py.File(os.path.join(model_folder, '{}/LENS/LENS{}/TARGET.hdf5'.format(tag, n + 1)), 'r') as file:
                target_lens = file['target'][...]
            
            value = numpy.zeros(bin_lens_size, dtype=numpy.float32)
            for m in range(bin_lens_size):
                value[m] = numpy.sum(target_lens[m, :]) / area / 3600
            density_lens.append(value)
            
            # Source
            with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/TARGET.hdf5'.format(tag, n + 1)), 'r') as file:
                target_source = file['target'][...]
            
            value = numpy.zeros(bin_source_size, dtype=numpy.float32)
            for m in range(bin_source_size):
                value[m] = numpy.sum(numpy.square(sigma0) / (numpy.square(sigma0) + numpy.square(application_sigma[target_source[m, :]]))) / area / 3600
            density_source.append(value)
        
        # Density
        density[tag]['LENS'] = numpy.mean(numpy.vstack(density_lens), axis=0).tolist()
        density[tag]['SOURCE'] = numpy.mean(numpy.vstack(density_source), axis=0).tolist()
    
    # Save
    with open(os.path.join(info_folder, 'DENSITY.json'), 'w') as file:
        json.dump(density, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Density')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)