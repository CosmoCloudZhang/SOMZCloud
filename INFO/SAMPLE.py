import os
import yaml
import h5py
import json
import time
import numpy
import argparse


def main(number, folder):
    '''
    Store the fiducial values of sample configuration
    
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
    sample = {'Y1': {}, 'Y10': {}}
    name_list = ['COPPER', 'GOLD', 'IRON', 'SILVER', 'TITANIUM', 'ZINC']
    
    # Loop
    for tag in sample.keys():
        print('Tag: {}'.format(tag))
        
        for name in name_list:
            print('Name: {}'.format(name))
            
            # Sample
            sample_lens = []
            sample_source = []
            
            for index in range(number + 1):
                print('Index: {}'.format(index))
                
                # Bin
                with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
                    bin_lens = file['bin_lens'][...]
                    bin_source = file['bin_source'][...]
                
                bin_lens_size = len(bin_lens) - 1
                bin_source_size = len(bin_source) - 1
                
                # Application
                with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
                    application_sigma = file['morphology']['sigma'][...]
                
                # Lens
                with h5py.File(os.path.join(model_folder, '{}/LENS/LENS{}/TARGET.hdf5'.format(tag, index)), 'r') as file:
                    target_lens = file['target'][...]
                
                value = numpy.zeros(bin_lens_size, dtype=numpy.float32)
                for m in range(bin_lens_size):
                    value[m] = numpy.sum(target_lens[m, :]) / area / 3600
                sample_lens.append(value)
                
                # Source
                with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/TARGET.hdf5'.format(tag, index)), 'r') as file:
                    target_source = file['target'][...]
                
                value = numpy.zeros(bin_source_size, dtype=numpy.float32)
                for m in range(bin_source_size):
                    value[m] = numpy.sum(numpy.square(sigma0) / (numpy.square(sigma0) + numpy.square(application_sigma[target_source[m, :]]))) / area / 3600
                sample_source.append(value)
            
            # Sample
            sample[tag][name] = {}
            sample[tag][name]['LENS'] = list(numpy.mean(numpy.vstack(sample_lens), axis=0))
            sample[tag][name]['SOURCE'] = list(numpy.mean(numpy.vstack(sample_source), axis=0))
        
    # Save
    with open(os.path.join(info_folder, 'SAMPLE.json'), 'w') as file:
        json.dump(sample, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Sample')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)