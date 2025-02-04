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
            degradation_dataset['meta']['count'] = file['meta']['count'][:].astype(numpy.int32)
            
            degradation_dataset['meta']['label'] = file['meta']['label'][:].astype(numpy.int32)
            degradation_dataset['meta']['coordinate1'] = file['meta']['coordinate1'][:].astype(numpy.int32)
            degradation_dataset['meta']['coordinate2'] = file['meta']['coordinate2'][:].astype(numpy.int32)
            
            degradation_dataset['meta']['size'] = file['meta']['size'][...]
            degradation_dataset['meta']['fraction'] = file['meta']['fraction'][...]
            degradation_dataset['meta']['redshift'] = file['meta']['redshift'][...]
            degradation_dataset['meta']['magnitude'] = file['meta']['magnitude'][...]
            
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
            augmentation_dataset['meta']['count'] = file['meta']['count'][:].astype(numpy.int32)
            
            augmentation_dataset['meta']['label'] = file['meta']['label'][:].astype(numpy.int32)
            augmentation_dataset['meta']['coordinate1'] = file['meta']['coordinate1'][:].astype(numpy.int32)
            augmentation_dataset['meta']['coordinate2'] = file['meta']['coordinate2'][:].astype(numpy.int32)
            
            augmentation_dataset['meta']['size'] = file['meta']['size'][...]
            augmentation_dataset['meta']['fraction'] = file['meta']['fraction'][...]
            augmentation_dataset['meta']['redshift'] = file['meta']['redshift'][...]
            augmentation_dataset['meta']['magnitude'] = file['meta']['magnitude'][...]
            
            augmentation_dataset['morphology'] = {key: file['morphology'][key][:].astype(numpy.float32) for key in file['morphology'].keys()}
            augmentation_dataset['photometry'] = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        
        # Combine
        degradation_sum = degradation_dataset['meta']['mean'] * degradation_dataset['meta']['count']
        augmentation_sum = augmentation_dataset['meta']['mean'] * augmentation_dataset['meta']['count']
        
        count = degradation_dataset['meta']['count'] + augmentation_dataset['meta']['count']
        mean = numpy.divide(degradation_sum + augmentation_sum, count, out=numpy.ones_like(count) * numpy.nan, where=count != 0)
        
        # Save
        os.makedirs(os.path.join(dataset_folder, '{}/'.format(tag)), exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, '{}/COMBINATION/'.format(tag)), exist_ok=True)
        
        with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('meta')
            file['meta'].create_dataset('mean', data=mean, dtype=numpy.float32)
            file['meta'].create_dataset('count', data=count, dtype=numpy.int32)
            
            file['meta'].create_dataset('label', data=numpy.append(degradation_dataset['meta']['label'], augmentation_dataset['meta']['label'], axis=0), dtype=numpy.int32)
            file['meta'].create_dataset('coordinate1', data=numpy.append(degradation_dataset['meta']['coordinate1'], augmentation_dataset['meta']['coordinate1'], axis=0), dtype=numpy.int32)
            file['meta'].create_dataset('coordinate2', data=numpy.append(degradation_dataset['meta']['coordinate2'], augmentation_dataset['meta']['coordinate2'], axis=0), dtype=numpy.int32)
            
            file['meta'].create_dataset('size1', data=degradation_dataset['meta']['size'], dtype=numpy.int32)
            file['meta'].create_dataset('fraction1', data=degradation_dataset['meta']['fraction'], dtype=numpy.float32)
            file['meta'].create_dataset('redshift1', data=degradation_dataset['meta']['redshift'], dtype=numpy.float32)
            file['meta'].create_dataset('magnitude1', data=degradation_dataset['meta']['magnitude'], dtype=numpy.float32)
            
            file['meta'].create_dataset('size2', data=augmentation_dataset['meta']['size'], dtype=numpy.int32)
            file['meta'].create_dataset('fraction2', data=augmentation_dataset['meta']['fraction'], dtype=numpy.float32)
            file['meta'].create_dataset('redshift2', data=augmentation_dataset['meta']['redshift'], dtype=numpy.float32)
            file['meta'].create_dataset('magnitude2', data=augmentation_dataset['meta']['magnitude'], dtype=numpy.float32)
            
            file.create_group('morphology')
            for key in degradation_dataset['morphology'].keys():
                file['morphology'].create_dataset(key, data=numpy.append(degradation_dataset['morphology'][key], augmentation_dataset['morphology'][key], axis=0), dtype=numpy.float32)
            
            file.create_group('photometry')
            for key in degradation_dataset['photometry'].keys():
                file['photometry'].create_dataset(key, data=numpy.append(degradation_dataset['photometry'][key], augmentation_dataset['photometry'][key], axis=0), dtype=numpy.float32)
    
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