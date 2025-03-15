import os
import yaml
import h5py
import json
import time
import numpy
import argparse


def main(number, folder):
    '''
    Store the fiducial values of survey configuration
    
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
    
    # Survey
    survey = {
        'Y1': {
            'AREA': 17936, 
        }, 
        'Y10': {
            'AREA': 17760,
        }
    }
    
    # Definition
    sigma0 = 0.26
    sky = 4 * numpy.pi * numpy.square(180 / numpy.pi)
    
    # Loop
    for tag in survey.keys():
        print('Tag: {}'.format(tag))
        
        # Bin
        with h5py.File(os.path.join(model_folder, '{}/SELECT/DATA0.hdf5'.format(tag)), 'r') as file:
            bin_lens_size = len(file['bin_lens'][...]) - 1
            bin_source_size = len(file['bin_source'][...]) - 1
        
        density_lens = numpy.zeros((bin_lens_size, number))
        density_source = numpy.zeros((bin_source_size, number))
        
        for index in range(1, number + 1):
            print('Index: {}'.format(index))
            
            # Application
            with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
                application_sigma = file['morphology']['sigma'][...]
            
            # Lens
            with h5py.File(os.path.join(model_folder, '{}/LENS/LENS{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
                select_lens = file['select'][...]
            
            for m in range(bin_lens_size):
                select = select_lens[m, :]
                density_lens[m, index - 1] = numpy.sum(select) / area / 3600
            
            # Source
            with h5py.File(os.path.join(model_folder, '{}/SOURCE/SOURCE{}/SELECT.hdf5'.format(tag, index)), 'r') as file:
                select_source = file['select'][...]
            
            for m in range(bin_source_size):
                select = select_source[m, :]
                density_source[m, index - 1] = numpy.sum(numpy.square(sigma0) / (numpy.square(application_sigma[select]) + numpy.square(sigma0))) / area / 3600
        
        survey[tag]['FRACTION'] = survey[tag]['AREA'] / sky
        survey[tag]['DENSITY_LENS'] = list(numpy.max(density_lens, axis=1))
        survey[tag]['DENSITY_SOURCE'] = list(numpy.max(density_source, axis=1))
    
    # Save
    with open(os.path.join(info_folder, 'SURVEY.json'), 'w') as file:
        json.dump(survey, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Survey')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)