import os
import h5py
import time
import numpy
import argparse


def main(tag, index, folder):
    '''
    Combine the datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the combination datasets
        folder (str): The base folder of the combination datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Selection
    selection_dataset = {
        'meta': {},
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        
        cell_size = file['meta']['cell_size'][...]
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        
        selection_dataset['meta']['size'] = file['meta']['size'][...]
        selection_dataset['meta']['amount'] = file['meta']['amount'][...]
        selection_dataset['meta']['sequence'] = file['meta']['sequence'][...]
        
        selection_dataset['meta']['cell_id'] = file['meta']['cell_id'][...]
        selection_dataset['meta']['cell_coordinate1'] = file['meta']['cell_coordinate1'][...]
        selection_dataset['meta']['cell_coordinate2'] = file['meta']['cell_coordinate2'][...]
        
        selection_dataset['meta']['cell_count'] = file['meta']['cell_count'][...]
        selection_dataset['meta']['cell_z_true'] = numpy.nan_to_num(file['meta']['cell_z_true'][...])
        
        selection_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        selection_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    
    # Augmentation
    augmentation_dataset = {
        'meta': {},
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        
        cell_size = file['meta']['cell_size'][...]
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        
        augmentation_dataset['meta']['size'] = file['meta']['size'][...]
        augmentation_dataset['meta']['fraction'] = file['meta']['fraction'][...]
        augmentation_dataset['meta']['redshift'] = file['meta']['redshift'][...]
        augmentation_dataset['meta']['magnitude'] = file['meta']['magnitude'][...]
        
        augmentation_dataset['meta']['cell_id'] = file['meta']['cell_id'][...]
        augmentation_dataset['meta']['cell_coordinate1'] = file['meta']['cell_coordinate1'][...]
        augmentation_dataset['meta']['cell_coordinate2'] = file['meta']['cell_coordinate2'][...]
        
        augmentation_dataset['meta']['cell_count'] = file['meta']['cell_count'][...]
        augmentation_dataset['meta']['cell_z_true'] = numpy.nan_to_num(file['meta']['cell_z_true'][...])
        
        augmentation_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        augmentation_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    
    # Combine
    cell_id = numpy.append(selection_dataset['meta']['cell_id'], augmentation_dataset['meta']['cell_id'], axis=0)
    cell_coordinate1 = numpy.append(selection_dataset['meta']['cell_coordinate1'], augmentation_dataset['meta']['cell_coordinate1'], axis=0)
    cell_coordinate2 = numpy.append(selection_dataset['meta']['cell_coordinate2'], augmentation_dataset['meta']['cell_coordinate2'], axis=0)
    
    selection_summation = selection_dataset['meta']['cell_z_true'] * selection_dataset['meta']['cell_count']
    augmentation_summation = augmentation_dataset['meta']['cell_z_true'] * augmentation_dataset['meta']['cell_count']
    
    size = selection_dataset['meta']['size'] + augmentation_dataset['meta']['size']
    cell_count = selection_dataset['meta']['cell_count'] + augmentation_dataset['meta']['cell_count']
    cell_z_true = numpy.divide(selection_summation + augmentation_summation, cell_count, out=numpy.ones_like(cell_count) * numpy.nan, where=cell_count != 0)
    
    # Save
    os.makedirs(os.path.join(dataset_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/COMBINATION/'.format(tag)), exist_ok=True)
    
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('size', data=size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size', data=cell_size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size1', data=cell_size1, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size2', data=cell_size2, dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_id', data=cell_id, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate1', data=cell_coordinate1, dtype=numpy.float32)
        file['meta'].create_dataset('cell_coordinate2', data=cell_coordinate2, dtype=numpy.float32)
        
        file['meta'].create_dataset('cell_count', data=cell_count, dtype=numpy.int32)
        file['meta'].create_dataset('cell_z_true', data=cell_z_true, dtype=numpy.float32)
        
        file.create_group('morphology')
        for key in selection_dataset['morphology'].keys():
            value = numpy.append(selection_dataset['morphology'][key], augmentation_dataset['morphology'][key], axis=0)
            file['morphology'].create_dataset(key, data=value, dtype=value.dtype)
        
        file.create_group('photometry')
        for key in selection_dataset['photometry'].keys():
            value = numpy.append(selection_dataset['photometry'][key], augmentation_dataset['photometry'][key], axis=0)
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
    PARSE.add_argument('--index', type=int, required=True, help='The index of the combination datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the combination datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)