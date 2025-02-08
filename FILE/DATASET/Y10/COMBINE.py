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
            
            degradation_dataset['meta']['size'] = file['meta']['size'][...]
            degradation_dataset['meta']['fraction'] = file['meta']['fraction'][...]
            degradation_dataset['meta']['redshift'] = file['meta']['redshift'][...]
            degradation_dataset['meta']['magnitude'] = file['meta']['magnitude'][...]
            
            degradation_dataset['meta']['cell_size1'] = file['meta']['cell_size1'][...]
            degradation_dataset['meta']['cell_size2'] = file['meta']['cell_size2'][...]
            
            degradation_dataset['meta']['cell_id'] = file['meta']['cell_id'][...]
            degradation_dataset['meta']['cell_coordinate1'] = file['meta']['cell_coordinate1'][...]
            degradation_dataset['meta']['cell_coordinate2'] = file['meta']['cell_coordinate2'][...]
            
            degradation_dataset['meta']['cell_count'] = file['meta']['cell_count'][...]
            degradation_dataset['meta']['cell_mean'] = numpy.nan_to_num(file['meta']['cell_mean'][...])
            
            degradation_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
            degradation_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
        
        # Augmentation
        augmentation_dataset = {
            'meta': {},
            'morphology': {},
            'photometry': {}
        }
        
        with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            augmentation_dataset['meta']['size'] = file['meta']['size'][...]
            augmentation_dataset['meta']['fraction'] = file['meta']['fraction'][...]
            augmentation_dataset['meta']['redshift'] = file['meta']['redshift'][...]
            augmentation_dataset['meta']['magnitude'] = file['meta']['magnitude'][...]
            
            augmentation_dataset['meta']['cell_size1'] = file['meta']['cell_size1'][...]
            augmentation_dataset['meta']['cell_size2'] = file['meta']['cell_size2'][...]
            
            augmentation_dataset['meta']['cell_id'] = file['meta']['cell_id'][...]
            augmentation_dataset['meta']['cell_coordinate1'] = file['meta']['cell_coordinate1'][...]
            augmentation_dataset['meta']['cell_coordinate2'] = file['meta']['cell_coordinate2'][...]
            
            augmentation_dataset['meta']['cell_mean'] = numpy.nan_to_num(file['meta']['cell_mean'][...])
            augmentation_dataset['meta']['cell_count'] = file['meta']['cell_count'][...]
            
            augmentation_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
            augmentation_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
        
        # Combine
        cell_size1 = numpy.average([degradation_dataset['meta']['cell_size1'], augmentation_dataset['meta']['cell_size1']], axis=0)
        cell_size2 = numpy.average([degradation_dataset['meta']['cell_size2'], augmentation_dataset['meta']['cell_size2']], axis=0)
        
        cell_id = numpy.append(degradation_dataset['meta']['cell_id'], augmentation_dataset['meta']['cell_id'], axis=0)
        cell_coordinate1 = numpy.append(degradation_dataset['meta']['cell_coordinate1'], augmentation_dataset['meta']['cell_coordinate1'], axis=0)
        cell_coordinate2 = numpy.append(degradation_dataset['meta']['cell_coordinate2'], augmentation_dataset['meta']['cell_coordinate2'], axis=0)
        
        degradation_summation = degradation_dataset['meta']['cell_mean'] * degradation_dataset['meta']['cell_count']
        augmentation_summation = augmentation_dataset['meta']['cell_mean'] * augmentation_dataset['meta']['cell_count']
        
        cell_count = degradation_dataset['meta']['cell_count'] + augmentation_dataset['meta']['cell_count']
        cell_mean = numpy.divide(degradation_summation + augmentation_summation, cell_count, out=numpy.ones_like(cell_count) * numpy.nan, where=cell_count != 0)
        
        # Save
        os.makedirs(os.path.join(dataset_folder, '{}/'.format(tag)), exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, '{}/COMBINATION/'.format(tag)), exist_ok=True)
        
        with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('meta')
            
            file['meta'].create_dataset('degradation_size', data=degradation_dataset['meta']['size'], dtype=numpy.int32)
            file['meta'].create_dataset('degradation_fraction', data=degradation_dataset['meta']['fraction'], dtype=numpy.float32)
            file['meta'].create_dataset('degradation_redshift', data=degradation_dataset['meta']['redshift'], dtype=numpy.float32)
            file['meta'].create_dataset('degradation_magnitude', data=degradation_dataset['meta']['magnitude'], dtype=numpy.float32)
            
            file['meta'].create_dataset('augmentation_size', data=augmentation_dataset['meta']['size'], dtype=numpy.int32)
            file['meta'].create_dataset('augmentation_fraction', data=augmentation_dataset['meta']['fraction'], dtype=numpy.float32)
            file['meta'].create_dataset('augmentation_redshift', data=augmentation_dataset['meta']['redshift'], dtype=numpy.float32)
            file['meta'].create_dataset('augmentation_magnitude', data=augmentation_dataset['meta']['magnitude'], dtype=numpy.float32)
            
            file['meta'].create_dataset('cell_size1', data=cell_size1, dtype=numpy.int32)
            file['meta'].create_dataset('cell_size2', data=cell_size2, dtype=numpy.int32)
            
            file['meta'].create_dataset('cell_id', data=cell_id, dtype=numpy.int32)
            file['meta'].create_dataset('cell_coordinate1', data=cell_coordinate1, dtype=numpy.float32)
            file['meta'].create_dataset('cell_coordinate2', data=cell_coordinate2, dtype=numpy.float32)
            
            file['meta'].create_dataset('cell_count', data=cell_count, dtype=numpy.int32)
            file['meta'].create_dataset('cell_mean', data=cell_mean, dtype=numpy.float32)
            
            file.create_group('morphology')
            for key in degradation_dataset['morphology'].keys():
                value = numpy.append(degradation_dataset['morphology'][key], augmentation_dataset['morphology'][key], axis=0)
                file['morphology'].create_dataset(key, data=value, dtype=value.dtype)
            
            file.create_group('photometry')
            for key in degradation_dataset['photometry'].keys():
                value = numpy.append(degradation_dataset['photometry'][key], augmentation_dataset['photometry'][key], axis=0)
                file['photometry'].create_dataset(key, data=value, dtype=value.dtype)
    
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