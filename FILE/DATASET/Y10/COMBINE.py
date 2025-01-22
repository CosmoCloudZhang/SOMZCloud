import os
import h5py
import time
import numpy
import argparse

def main(tag, number, folder):
    '''
    Combine the datasets
    
    Arguments:
        tag (str): The tag of the configuration
        number (int): The number of the combination datasets
        folder (str): The base folder of the combination datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    for index in range(1, number + 1):
        print('Index: {}'.format(index))
        
        # Degradation
        degradation_dataset = {
            'meta': {},
            'morphology': {},
            'photometry': {}
        }
        
        with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            degradation_dataset['meta']['mean'] = numpy.nan_to_num(file['meta']['mean'][:].astype(numpy.float32))
            degradation_dataset['meta']['occupation'] = file['meta']['occupation'][:].astype(numpy.float32)
            
            degradation_dataset['meta']['label'] = file['meta']['label'][:].astype(numpy.float32)
            degradation_dataset['meta']['coordinate1'] = file['meta']['coordinate1'][:].astype(numpy.float32)
            degradation_dataset['meta']['coordinate2'] = file['meta']['coordinate2'][:].astype(numpy.float32)
            
            degradation_dataset['morphology'] = {key: file['morphology'][key][:].astype(numpy.float32) for key in file['morphology'].keys()}
            degradation_dataset['photometry'] = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        
        # Augmentation
        augmentation_dataset = {
            'meta': {},
            'morphology': {},
            'photometry': {}
        }
        
        with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            augmentation_dataset['meta']['mean'] = numpy.nan_to_num(file['meta']['mean'][:].astype(numpy.float32))
            augmentation_dataset['meta']['occupation'] = file['meta']['occupation'][:].astype(numpy.float32)
            
            augmentation_dataset['meta']['label'] = file['meta']['label'][:].astype(numpy.float32)
            augmentation_dataset['meta']['coordinate1'] = file['meta']['coordinate1'][:].astype(numpy.float32)
            augmentation_dataset['meta']['coordinate2'] = file['meta']['coordinate2'][:].astype(numpy.float32)
            
            augmentation_dataset['morphology'] = {key: file['morphology'][key][:].astype(numpy.float32) for key in file['morphology'].keys()}
            augmentation_dataset['photometry'] = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        
        # Combine
        degradation_sum = degradation_dataset['meta']['mean'] * degradation_dataset['meta']['occupation']
        augmentation_sum = augmentation_dataset['meta']['mean'] * augmentation_dataset['meta']['occupation']
        
        occupation = degradation_dataset['meta']['occupation'] + augmentation_dataset['meta']['occupation']
        mean = numpy.divide(degradation_sum + augmentation_sum, occupation, out=numpy.ones_like(occupation) * numpy.nan, where=occupation != 0)
        
        # Save
        os.makedirs(os.path.join(dataset_folder, '{}/'.format(tag)), exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, '{}/COMBINATION/'.format(tag)), exist_ok=True)
        
        with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('meta')
            file['meta'].create_dataset('mean', data=mean)
            file['meta'].create_dataset('occupation', data=occupation)
            
            file['meta'].create_dataset('label', data=numpy.append(degradation_dataset['meta']['label'], augmentation_dataset['meta']['label'], axis=0))
            file['meta'].create_dataset('coordinate1', data=numpy.append(degradation_dataset['meta']['coordinate1'], augmentation_dataset['meta']['coordinate1'], axis=0))
            file['meta'].create_dataset('coordinate2', data=numpy.append(degradation_dataset['meta']['coordinate2'], augmentation_dataset['meta']['coordinate2'], axis=0))
            
            file.create_group('morphology')
            for key in degradation_dataset['morphology'].keys():
                file['morphology'].create_dataset(key, data=numpy.append(degradation_dataset['morphology'][key], augmentation_dataset['morphology'][key], axis=0))
            
            file.create_group('photometry')
            for key in degradation_dataset['photometry'].keys():
                file['photometry'].create_dataset(key, data=numpy.append(degradation_dataset['photometry'][key], augmentation_dataset['photometry'][key], axis=0))
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Combination Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the combination datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the combination datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)