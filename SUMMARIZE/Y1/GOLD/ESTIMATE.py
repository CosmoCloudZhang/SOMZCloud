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
    compare_folder = os.path.join(folder, 'COMPARE/')
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
    with h5py.File(os.path.join(compare_folder, '{}/TARGET/DATA{}.hdf5'.format(tag, index)), 'r') as file:
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
    
    # Degradation
    degradation_dataset = {
        'meta': {},
        'morphology': {}, 
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_dataset['meta'] = {key: file['meta'][key][...] for key in file['meta'].keys()}
        degradation_dataset['morphology'] = {key: file['morphology'][key][...] for key in file['morphology'].keys()}
        degradation_dataset['photometry'] = {key: file['photometry'][key][...] for key in file['photometry'].keys()}
    degradation_z_spec = degradation_dataset['photometry']['redshift'][...]
    
    # Reference
    with h5py.File(os.path.join(compare_folder, '{}/REFERENCE/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_z_phot = file['z_phot'][...]
    
    # Loop
    chunk = 100000
    degradation_size = len(degradation_z_spec)
    degradation_cell_coordinate = numpy.zeros((degradation_size, 2), dtype=numpy.int32)
    
    for k in range(degradation_size // chunk + 1):
        begin = k * chunk
        stop = min((k + 1) * chunk, degradation_size)
        
        if begin < stop:
            degradation = {key: degradation_dataset['photometry'][key][begin: stop] for key in model['usecols']}
            
            degradation_column = somoclu_som._computemagcolordata(data=degradation, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            degradation_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], degradation_column)
    
    degradation_cell_coordinate1 = degradation_cell_coordinate[:, 0]
    degradation_cell_coordinate2 = degradation_cell_coordinate[:, 1]
    degradation_cell_id = numpy.ravel_multi_index(numpy.transpose(degradation_cell_coordinate), (model['n_rows'], model['n_columns']))
    
    degradation_cell_size = model['n_rows'] * model['n_columns']
    degradation_cell_count = numpy.bincount(degradation_cell_id, minlength=degradation_cell_size)
    
    degradation_cell_z_phot = numpy.bincount(degradation_cell_id, weights=degradation_z_phot, minlength=degradation_cell_size)
    degradation_cell_z_phot = numpy.divide(degradation_cell_z_phot, degradation_cell_count, out=numpy.ones(degradation_cell_size) * numpy.nan, where=degradation_cell_count != 0)
    
    degradation_cell_z_spec = numpy.bincount(degradation_cell_id, weights=degradation_z_spec, minlength=degradation_cell_size)
    degradation_cell_z_spec = numpy.divide(degradation_cell_z_spec, degradation_cell_count, out=numpy.ones(degradation_cell_size) * numpy.nan, where=degradation_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(summarize_folder, '{}/{}/ESTIMATE/ESTIMATE{}.hdf5'.format(tag, name, index)), 'a') as file:
        file.create_group('degradation')
        
        file['degradation'].create_dataset('size', data=degradation_size, dtype=numpy.int32)
        file['degradation'].create_dataset('cell_size', data=degradation_cell_size, dtype=numpy.int32)
        file['degradation'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['degradation'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['degradation'].create_dataset('cell_id', data=degradation_cell_id, dtype=numpy.int32)
        file['degradation'].create_dataset('cell_coordinate1', data=degradation_cell_coordinate1, dtype=numpy.int32)
        file['degradation'].create_dataset('cell_coordinate2', data=degradation_cell_coordinate2, dtype=numpy.int32)
        
        file['degradation'].create_dataset('cell_count', data=degradation_cell_count, dtype=numpy.int32)
        file['degradation'].create_dataset('cell_z_phot', data=degradation_cell_z_phot, dtype=numpy.float32)
        file['degradation'].create_dataset('cell_z_spec', data=degradation_cell_z_spec, dtype=numpy.float32)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Summarize Gold Estimate Lens')
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