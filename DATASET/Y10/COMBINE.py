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
    random_generator = numpy.random.default_rng(seed=index)
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, '{}/'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/COMBINATION/'.format(tag)), exist_ok=True)
    
    # Load
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_cell_count = file['meta']['cell_count'][...]
    
    # Degradation
    degradation_dataset = {
        'meta': {},
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        
        cell_size = file['meta']['cell_size'][...]
        cell_size1 = file['meta']['cell_size1'][...]
        cell_size2 = file['meta']['cell_size2'][...]
        
        degradation_dataset['meta']['size'] = file['meta']['size'][...]
        degradation_dataset['meta']['amount'] = file['meta']['amount'][...]
        degradation_dataset['meta']['sequence'] = file['meta']['sequence'][...]
        
        degradation_dataset['meta']['cell_id'] = file['meta']['cell_id'][...]
        degradation_dataset['meta']['cell_coordinate1'] = file['meta']['cell_coordinate1'][...]
        degradation_dataset['meta']['cell_coordinate2'] = file['meta']['cell_coordinate2'][...]
        
        degradation_dataset['meta']['cell_count'] = file['meta']['cell_count'][...]
        degradation_dataset['meta']['cell_z_true'] = numpy.nan_to_num(file['meta']['cell_z_true'][...])
        
        degradation_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        degradation_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    degradation_rank = numpy.zeros(degradation_dataset['meta']['size'], dtype=numpy.int32)
    
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
    augmentation_rank = numpy.ones(augmentation_dataset['meta']['size'], dtype=numpy.int32)
    
    # Combination
    combination_cell_id = numpy.append(degradation_dataset['meta']['cell_id'], augmentation_dataset['meta']['cell_id'], axis=0)
    combination_cell_coordinate1 = numpy.append(degradation_dataset['meta']['cell_coordinate1'], augmentation_dataset['meta']['cell_coordinate1'], axis=0)
    combination_cell_coordinate2 = numpy.append(degradation_dataset['meta']['cell_coordinate2'], augmentation_dataset['meta']['cell_coordinate2'], axis=0)
    
    combination_rank = numpy.concatenate([degradation_rank, augmentation_rank], axis=0)
    combination_size = degradation_dataset['meta']['size'] + augmentation_dataset['meta']['size']
    combination_cell_count = degradation_dataset['meta']['cell_count'] + augmentation_dataset['meta']['cell_count']
    
    # Sampling
    combination_weight = numpy.divide(application_cell_count, combination_cell_count, out=numpy.zeros(cell_size), where=combination_cell_count != 0)
    combination_probability = combination_weight[combination_cell_id] / numpy.sum(combination_weight[combination_cell_id])
    indices = random_generator.choice(combination_size, size=combination_size, replace=True, p=combination_probability)
    
    combination_cell_count = numpy.bincount(combination_cell_id[indices], minlength=cell_size)
    combination_redshift_true = numpy.append(degradation_dataset['photometry']['redshift_true'], augmentation_dataset['photometry']['redshift_true'], axis=0)
    combination_cell_z_true = numpy.divide(numpy.bincount(combination_cell_id[indices], weights=combination_redshift_true[indices], minlength=cell_size), combination_cell_count, out=numpy.ones(cell_size) * numpy.nan, where=combination_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('cell_size', data=cell_size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size1', data=cell_size1, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size2', data=cell_size2, dtype=numpy.int32)
        file['meta'].create_dataset('size', data=combination_size, dtype=numpy.int32)
        
        file['meta'].create_dataset('rank', data=combination_rank[indices], dtype=numpy.int32)
        file['meta'].create_dataset('cell_id', data=combination_cell_id[indices], dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate1', data=combination_cell_coordinate1[indices], dtype=numpy.float32)
        file['meta'].create_dataset('cell_coordinate2', data=combination_cell_coordinate2[indices], dtype=numpy.float32)
        
        file['meta'].create_dataset('cell_count', data=combination_cell_count, dtype=numpy.int32)
        file['meta'].create_dataset('cell_z_true', data=combination_cell_z_true, dtype=numpy.float32)
        
        file.create_group('morphology')
        for key in degradation_dataset['morphology'].keys():
            value = numpy.append(degradation_dataset['morphology'][key], augmentation_dataset['morphology'][key], axis=0)
            file['morphology'].create_dataset(key, data=value[indices], dtype=value.dtype)
        
        file.create_group('photometry')
        for key in degradation_dataset['photometry'].keys():
            value = numpy.append(degradation_dataset['photometry'][key], augmentation_dataset['photometry'][key], axis=0)
            file['photometry'].create_dataset(key, data=value[indices], dtype=value.dtype)
    
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