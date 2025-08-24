import os
import h5py
import time
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som


def main(tag, index, folder):
    '''
    Combine the datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the restriction datasets
        folder (str): The base folder of the restriction datasets
    
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
    os.makedirs(os.path.join(dataset_folder, '{}/RESTRICTION/'.format(tag)), exist_ok=True)
    
    # Association
    association_dataset = {
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/ASSOCIATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        association_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        association_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    
    # Combination
    combination_dataset = {
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        combination_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    
    association_size = len(association_dataset['photometry']['redshift'])
    combination_size = len(combination_dataset['photometry']['redshift'])
    indices = random_generator.choice(numpy.arange(association_size), size=combination_size, replace=True)
    
    # Restriction
    restriction_dataset = {
        'morphology': {key: association_dataset['morphology'][key][indices] for key in association_dataset['morphology'].keys()},
        'photometry': {key: association_dataset['photometry'][key][indices] for key in association_dataset['photometry'].keys()}
    }
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(dataset_folder, '{}/SOM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    restriction_size = len(restriction_dataset['photometry']['redshift'])
    restriction_cell_coordinate = numpy.zeros((restriction_size, 2), dtype=numpy.int32)
    
    for m in range(restriction_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, restriction_size)
        
        if begin < stop:
            restriction = {key: restriction_dataset['photometry'][key][begin: stop] for key in model['usecols']}
            
            restriction_column = somoclu_som._computemagcolordata(data=restriction, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            restriction_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], restriction_column)
    
    restriction_cell_coordinate1 = restriction_cell_coordinate[:, 0]
    restriction_cell_coordinate2 = restriction_cell_coordinate[:, 1]
    restriction_cell_id = numpy.ravel_multi_index(numpy.transpose(restriction_cell_coordinate), (model['n_rows'], model['n_columns']))
    
    cell_size = model['n_rows'] * model['n_columns']
    restriction_cell_count = numpy.bincount(restriction_cell_id, minlength=cell_size)
    
    restriction_cell_z_true = numpy.divide(numpy.bincount(restriction_cell_id, weights=restriction_dataset['photometry']['redshift_true'], minlength=cell_size), restriction_cell_count, out=numpy.ones(cell_size) * numpy.nan, where=restriction_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/RESTRICTION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('cell_size', data=cell_size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['meta'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_id', data=restriction_cell_id, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate1', data=restriction_cell_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate2', data=restriction_cell_coordinate2, dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_count', data=restriction_cell_count, dtype=numpy.int32)
        file['meta'].create_dataset('cell_z_true', data=restriction_cell_z_true, dtype=numpy.float32)
        
        file.create_group('morphology')
        for key in restriction_dataset['morphology'].keys():
            file['morphology'].create_dataset(key, data=restriction_dataset['morphology'][key], dtype=restriction_dataset['morphology'][key].dtype)
        
        file.create_group('photometry')
        for key in restriction_dataset['photometry'].keys():
            file['photometry'].create_dataset(key, data=restriction_dataset['photometry'][key], dtype=restriction_dataset['photometry'][key].dtype)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Restriction Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the restriction datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the restriction datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)