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
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Load
    with open(os.path.join(dataset_folder, 'CATALOG/OBSERVE.yaml'), 'r') as file:
        observe = yaml.safe_load(file)
        observation_list = observe['healpix_pixels']
        area = observe['sky_area'] / len(observation_list) * len(observation_list) // 2
    
    # Definition
    metric = []
    sigma0 = 0.26
    for index in range(number + 1):
        print('Index: {}'.format(index))
        
        metric.append({'Y1': {}, 'Y10': {}})
        for tag in metric[index].keys():
            
            # Application
            with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
                application_sigma = file['morphology']['sigma'][...]
            
            metric[index][tag]['NUMBER'] = len(application_sigma)
            metric[index][tag]['NUMBER_DENSITY'] = len(application_sigma) / area / 3600
            metric[index][tag]['EFFECTIVE_NUMBER_DENSITY'] = numpy.sum(numpy.square(sigma0) / (numpy.square(sigma0) + numpy.square(application_sigma))) / area / 3600
    
    # Save
    with open(os.path.join(info_folder, 'METRIC.json'), 'w') as file:
        json.dump(metric, file, indent=4)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Info Metric')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the datasets')
    
    # Parse
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)