import os
import time
import h5py
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som


def main(tag, name, index, folder):
    '''
    Generate dataset file for the SOMoclu Estimator.
    
    Arguments:
        tag (str): The tag of configuration
        name (str): The name of configuration
        index (int): The index of all the datasets
        folder (str): The base folder of all the datasets
    
    Returns:
        duration (float): The duration of the process.
    '''
    # Data store
    start = time.time()
    print('Index: {}'.format(index))
    
    # Path
    model_folder = os.path.join(folder, 'MODEL/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    summarize_folder = os.path.join(folder, 'SUMMARIZE/')
    os.makedirs(os.path.join(summarize_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(summarize_folder, '{}/{}/ESTIMATE/'.format(tag, name)), exist_ok=True)
    
    # RAIL
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(summarize_folder, '{}/{}/INFORM/INFORM{}.pkl'.format(tag, name, index))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Application
    application_dataset = {
        'meta': {},
        'morphology': {}, 
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_dataset['meta'] = {key: file['meta'][key][...] for key in file['meta'].keys()}
        application_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        application_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    application_z_true = application_dataset['photometry']['redshift_true'][...]
    
    # Target
    with h5py.File(os.path.join(model_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_z_phot = file['z_phot'][...]
    
    # Loop
    chunk = 100000
    application_size = len(application_z_true)
    application_cell_coordinate = numpy.zeros((application_size, 2), dtype=numpy.int32)
    
    for k in range(application_size // chunk + 1):
        begin = k * chunk
        stop = min((k + 1) * chunk, application_size)
        
        if begin < stop:
            application = {key: application_dataset['photometry'][key][begin: stop] for key in model['usecols']}
            
            application_column = somoclu_som._computemagcolordata(data=application, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            application_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], application_column)
    
    application_cell_coordinate1 = application_cell_coordinate[:, 0]
    application_cell_coordinate2 = application_cell_coordinate[:, 1]
    application_cell_id = numpy.ravel_multi_index(numpy.transpose(application_cell_coordinate), (model['n_rows'], model['n_columns']))
    
    application_cell_size = model['n_rows'] * model['n_columns']
    application_cell_count = numpy.bincount(application_cell_id, minlength=application_cell_size)
    
    application_cell_z_phot = numpy.bincount(application_cell_id, weights=application_z_phot, minlength=application_cell_size)
    application_cell_z_phot = numpy.divide(application_cell_z_phot, application_cell_count, out=numpy.ones(application_cell_size) * numpy.nan, where=application_cell_count != 0)
    
    application_cell_z_true = numpy.bincount(application_cell_id, weights=application_z_true, minlength=application_cell_size)
    application_cell_z_true = numpy.divide(application_cell_z_true, application_cell_count, out=numpy.ones(application_cell_size) * numpy.nan, where=application_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(summarize_folder, '{}/{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, name, index)), 'w') as file:
        file.create_group('application')
        
        file['application'].create_dataset('size', data=application_size, dtype=numpy.int32)
        file['application'].create_dataset('cell_size', data=application_cell_size, dtype=numpy.int32)
        file['application'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['application'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['application'].create_dataset('cell_id', data=application_cell_id, dtype=numpy.int32)
        file['application'].create_dataset('cell_coordinate1', data=application_cell_coordinate1, dtype=numpy.int32)
        file['application'].create_dataset('cell_coordinate2', data=application_cell_coordinate2, dtype=numpy.int32)
        
        file['application'].create_dataset('cell_count', data=application_cell_count, dtype=numpy.int32)
        file['application'].create_dataset('cell_z_phot', data=application_cell_z_phot, dtype=numpy.float32)
        file['application'].create_dataset('cell_z_true', data=application_cell_z_true, dtype=numpy.float32)
    
    # Combination
    combination_dataset = {
        'meta': {},
        'morphology': {}, 
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/COMBINATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_dataset['meta'] = {key: file['meta'][key][...] for key in file['meta'].keys()}
        combination_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        combination_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    combination_z_spec = combination_dataset['photometry']['redshift'][...]
    
    # Reference
    with h5py.File(os.path.join(model_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        combination_z_phot = file['z_phot'][...]
    
    # Loop
    chunk = 100000
    combination_size = len(combination_z_spec)
    combination_cell_coordinate = numpy.zeros((combination_size, 2), dtype=numpy.int32)
    
    for k in range(combination_size // chunk + 1):
        begin = k * chunk
        stop = min((k + 1) * chunk, combination_size)
        
        if begin < stop:
            combination = {key: combination_dataset['photometry'][key][begin: stop] for key in model['usecols']}
            
            combination_column = somoclu_som._computemagcolordata(data=combination, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            combination_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], combination_column)
    
    combination_cell_coordinate1 = combination_cell_coordinate[:, 0]
    combination_cell_coordinate2 = combination_cell_coordinate[:, 1]
    combination_cell_id = numpy.ravel_multi_index(numpy.transpose(combination_cell_coordinate), (model['n_rows'], model['n_columns']))
    
    combination_cell_size = model['n_rows'] * model['n_columns']
    combination_cell_count = numpy.bincount(combination_cell_id, minlength=combination_cell_size)
    
    combination_cell_z_phot = numpy.bincount(combination_cell_id, weights=combination_z_phot, minlength=combination_cell_size)
    combination_cell_z_phot = numpy.divide(combination_cell_z_phot, combination_cell_count, out=numpy.ones(combination_cell_size) * numpy.nan, where=combination_cell_count != 0)
    
    combination_cell_z_spec = numpy.bincount(combination_cell_id, weights=combination_z_spec, minlength=combination_cell_size)
    combination_cell_z_spec = numpy.divide(combination_cell_z_spec, combination_cell_count, out=numpy.ones(combination_cell_size) * numpy.nan, where=combination_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(summarize_folder, '{}/{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, name, index)), 'a') as file:
        file.create_group('combination')
        
        file['combination'].create_dataset('size', data=combination_size, dtype=numpy.int32)
        file['combination'].create_dataset('cell_size', data=combination_cell_size, dtype=numpy.int32)
        file['combination'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['combination'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['combination'].create_dataset('cell_id', data=combination_cell_id, dtype=numpy.int32)
        file['combination'].create_dataset('cell_coordinate1', data=combination_cell_coordinate1, dtype=numpy.int32)
        file['combination'].create_dataset('cell_coordinate2', data=combination_cell_coordinate2, dtype=numpy.int32)
        
        file['combination'].create_dataset('cell_count', data=combination_cell_count, dtype=numpy.int32)
        file['combination'].create_dataset('cell_z_phot', data=combination_cell_z_phot, dtype=numpy.float32)
        file['combination'].create_dataset('cell_z_spec', data=combination_cell_z_spec, dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Copper Estimate')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of configuration')
    PARSE.add_argument('--name', type=str, required=True, help='The name of configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of all the datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of all the datasets')
    
    # Parse
    TAG = PARSE.parse_args().tag
    NAME = PARSE.parse_args().name
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NAME, INDEX, FOLDER)