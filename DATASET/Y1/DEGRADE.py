import os
import h5py
import time
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som


def main(tag, index, number, folder):
    '''
    Create the degradation datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the degradation datasets
        folder (str): The base folder of the degradation datasets
    
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
    os.makedirs(os.path.join(dataset_folder, '{}/DEGRADATION/'.format(tag)), exist_ok=True)
    
    # Amount
    amount1 = {'Y1': 4, 'Y10': 7}
    amount2 = {'Y1': 6, 'Y10': 12}
    amount = random_generator.integers(low=amount1[tag], high=amount2[tag], endpoint=True)
    
    # Sequence
    sequence = random_generator.integers(low=0, high=number, size=amount, endpoint=True)
    
    # Size
    size1 = {'Y1': 100000, 'Y10': 250000}
    size2 = {'Y1': 200000, 'Y10': 500000}
    size = random_generator.integers(low=size1[tag], high=size2[tag], endpoint=True)
    
    # Selection
    selection_dataset = {
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        selection_dataset['morphology'] = {key: numpy.array([]) for key in file['morphology'].keys()}
        selection_dataset['photometry'] = {key: numpy.array([]) for key in file['photometry'].keys()}
    
    for rank in sequence:
        with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, rank)), 'r') as file:
            for key in file['morphology'].keys():
                selection_dataset['morphology'][key] = numpy.append(selection_dataset['morphology'][key][...], file['morphology'][key][...])
            
            for key in file['photometry'].keys():
                selection_dataset['photometry'][key] = numpy.append(selection_dataset['photometry'][key][...], file['photometry'][key][...])
    
    selection_size = len(selection_dataset['photometry']['redshift'])
    indices = random_generator.choice(numpy.arange(selection_size), size=size, replace=True)
    
    # Degradation
    degradation_dataset = {
        'morphology': {key: selection_dataset['morphology'][key][indices] for key in selection_dataset['morphology'].keys()},
        'photometry': {key: selection_dataset['photometry'][key][indices] for key in selection_dataset['photometry'].keys()}
    }
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(dataset_folder, '{}/SOM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    degradation_size = len(degradation_dataset['photometry']['redshift'])
    degradation_cell_coordinate = numpy.zeros((degradation_size, 2), dtype=numpy.int32)
    
    for m in range(degradation_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, degradation_size)
        
        if begin < stop:
            degradation = {key: degradation_dataset['photometry'][key][begin: stop] for key in model['usecols']}
            
            degradation_column = somoclu_som._computemagcolordata(data=degradation, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            degradation_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], degradation_column)
    
    degradation_cell_coordinate1 = degradation_cell_coordinate[:, 0]
    degradation_cell_coordinate2 = degradation_cell_coordinate[:, 1]
    degradation_cell_id = numpy.ravel_multi_index(numpy.transpose(degradation_cell_coordinate), (model['n_rows'], model['n_columns']))
    
    cell_size = model['n_rows'] * model['n_columns']
    degradation_cell_count = numpy.bincount(degradation_cell_id, minlength=cell_size)
    degradation_cell_z_true = numpy.divide(numpy.bincount(degradation_cell_id, weights=degradation_dataset['photometry']['redshift_true'], minlength=cell_size), degradation_cell_count, out=numpy.ones(cell_size) * numpy.nan, where=degradation_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('size', data=size, dtype=numpy.int32)
        file['meta'].create_dataset('amount', data=amount, dtype=numpy.int32)
        file['meta'].create_dataset('sequence', data=sequence, dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_size', data=cell_size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['meta'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_id', data=degradation_cell_id, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate1', data=degradation_cell_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate2', data=degradation_cell_coordinate2, dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_count', data=degradation_cell_count, dtype=numpy.int32)
        file['meta'].create_dataset('cell_z_true', data=degradation_cell_z_true, dtype=numpy.float32)
        
        file.create_group('morphology')
        for key in degradation_dataset['morphology'].keys():
            file['morphology'].create_dataset(key, data=degradation_dataset['morphology'][key], dtype=degradation_dataset['morphology'][key].dtype)
        
        file.create_group('photometry')
        for key in degradation_dataset['photometry'].keys():
            file['photometry'].create_dataset(key, data=degradation_dataset['photometry'][key], dtype=degradation_dataset['photometry'][key].dtype)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Degradation datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the degradation datasets')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the degradation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the degradation datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, NUMBER, FOLDER)