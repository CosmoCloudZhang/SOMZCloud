import os
import h5py
import time
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som


def main(tag, index, folder):
    '''
    Create the augmentation datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the augmentation datasets
        folder (str): The base folder of the augmentation datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    print('Index: {}'.format(index))
    random_generator = numpy.random.default_rng(seed=index)
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/AUGMENTATION/'.format(tag)), exist_ok=True)
    
    # Application
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size = file['meta']['cell_size'][...]
        application_cell_count = file['meta']['cell_count'][...]
    application_size = numpy.sum(application_cell_count)
    
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        cell_size = file['meta']['cell_size'][...]
        degradation_cell_count = file['meta']['cell_count'][...]
        degradation_redshift = file['photometry']['redshift'][...]
        degradation_magnitude = file['photometry']['mag_i_lsst'][...]
    degradation_size = len(degradation_redshift)
    
    # Association
    association_dataset = {
        'meta': {},
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/ASSOCIATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        association_dataset['meta'] = {key: file['meta'][key][...] for key in file['meta'].keys()}
        association_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        association_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    
    association_cell_id = association_dataset['meta']['cell_id']
    association_size = len(association_dataset['photometry']['redshift'])
    filter = numpy.isin(association_cell_id, numpy.arange(cell_size)[degradation_cell_count == 0])
    
    # Magnitude
    magnitude = numpy.max(degradation_magnitude)
    filter = filter | (association_dataset['photometry']['mag_i_lsst'] > magnitude)
    
    # Redshift
    redshift = numpy.max(degradation_redshift)
    filter = filter | (association_dataset['photometry']['redshift'] > redshift)
    
    # Fraction
    fraction = numpy.sum(application_cell_count[degradation_cell_count == 0]) / application_size
    
    # Size
    size = int(degradation_size * fraction * (1 + fraction))
    indices = random_generator.choice(numpy.arange(association_size)[filter], size=size, replace=True)
    
    # Augmentation
    augmentation_dataset = {
        'morphology': {key: association_dataset['morphology'][key][indices] for key in association_dataset['morphology'].keys()},
        'photometry': {key: association_dataset['photometry'][key][indices] for key in association_dataset['photometry'].keys()}
    }
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(dataset_folder, '{}/SOM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    augmentation_size = len(augmentation_dataset['photometry']['redshift'])
    augmentation_cell_coordinate = numpy.zeros((augmentation_size, 2), dtype=numpy.int32)
    
    for m in range(augmentation_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, augmentation_size)
        
        if begin < stop:
            augmentation = {key: augmentation_dataset['photometry'][key][begin: stop] for key in model['usecols']}
            
            augmentation_column = somoclu_som._computemagcolordata(data=augmentation, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            augmentation_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], augmentation_column)
    
    augmentation_cell_coordinate1 = augmentation_cell_coordinate[:, 0]
    augmentation_cell_coordinate2 = augmentation_cell_coordinate[:, 1]
    augmentation_cell_id = numpy.ravel_multi_index(numpy.transpose(augmentation_cell_coordinate), dims=(model['n_rows'], model['n_columns']))
    
    cell_size = model['n_rows'] * model['n_columns']
    augmentation_cell_count = numpy.bincount(augmentation_cell_id, minlength=cell_size)
    augmentation_cell_z_true = numpy.divide(numpy.bincount(augmentation_cell_id, weights=augmentation_dataset['photometry']['redshift_true'], minlength=cell_size), augmentation_cell_count, out=numpy.ones(cell_size) * numpy.nan, where=augmentation_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('size', data=size, dtype=numpy.int32)
        file['meta'].create_dataset('fraction', data=fraction, dtype=numpy.float32)
        file['meta'].create_dataset('redshift', data=redshift, dtype=numpy.float32)
        file['meta'].create_dataset('magnitude', data=magnitude, dtype=numpy.float32)
        
        file['meta'].create_dataset('cell_size', data=cell_size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['meta'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_id', data=augmentation_cell_id, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate1', data=augmentation_cell_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate2', data=augmentation_cell_coordinate2, dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_count', data=augmentation_cell_count, dtype=numpy.int32)
        file['meta'].create_dataset('cell_z_true', data=augmentation_cell_z_true, dtype=numpy.float32)
        
        file.create_group('morphology')
        for key in association_dataset['morphology'].keys():
            file['morphology'].create_dataset(key, data=augmentation_dataset['morphology'][key], dtype=augmentation_dataset['morphology'][key].dtype)
        
        file.create_group('photometry')
        for key in association_dataset['photometry'].keys():
            file['photometry'].create_dataset(key, data=augmentation_dataset['photometry'][key], dtype=augmentation_dataset['photometry'][key].dtype)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the augmentation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the augmentation datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)