import os
import h5py
import time
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som


def main(tag, index, folder):
    '''
    Create the selection datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the selection datasets
        folder (str): The base folder of the selection datasets
    
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
    os.makedirs(os.path.join(dataset_folder, '{}/SELECTION/'.format(tag)), exist_ok=True)
    
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
    
    # Magnitude
    magnitude1 = {'Y1': 20.0, 'Y10': 20.0}
    magnitude2 = {'Y1': 24.0, 'Y10': 25.0}
    magnitude = random_generator.uniform(low=magnitude1[tag], high=magnitude2[tag])
    
    filter = (application_dataset['photometry']['mag_i_lsst'] < magnitude)
    
    # Redshift
    redshift1 = {'Y1': 0.5, 'Y10': 0.5}
    redshift2 = {'Y1': 2.5, 'Y10': 3.0}
    redshift = random_generator.uniform(low=redshift1[tag], high=redshift2[tag])
    
    filter = filter & (application_dataset['photometry']['redshift'] < redshift)
    
    # Color
    color1 = {'Y1': 1.0, 'Y10': 1.0}
    color2 = {'Y1': 3.0, 'Y10': 3.0}
    color = random_generator.uniform(low=color1[tag], high=color2[tag])
    
    # Angle
    angle1 = {'Y1': 0.0, 'Y10': 0.0}
    angle2 = {'Y1': numpy.pi, 'Y10': numpy.pi}
    angle = random_generator.uniform(low=angle1[tag], high=angle2[tag])
    
    application_color = application_dataset['photometry']['mag_g_lsst'] - application_dataset['photometry']['mag_z_lsst']
    filter = filter & (application_dataset['photometry']['mag_i_lsst'] - magnitude  - numpy.tan(angle) * (application_color - color) < 0)
    
    # Fraction
    fraction1 = {'Y1': 0.2, 'Y10': 0.2}
    fraction2 = {'Y1': 0.8, 'Y10': 0.8}
    fraction = random_generator.uniform(low=fraction1[tag], high=fraction2[tag])
    
    # Cell
    application_cell_id = application_dataset['meta']['cell_id']
    application_cell_size = application_dataset['meta']['cell_size']
    application_cell_id_select = random_generator.choice(numpy.arange(application_cell_size), size=int(application_cell_size * fraction), replace=False)
    
    filter = filter & numpy.isin(application_cell_id, application_cell_id_select)
    
    # Factor
    factor1 = {'Y1': 0.5, 'Y10': 0.5}
    factor2 = {'Y1': 5.0, 'Y10': 5.0}
    factor = random_generator.uniform(low=factor1[tag], high=factor2[tag])
    rate = 1 / (1 + factor * numpy.exp(application_dataset['photometry']['mag_i_lsst'] - magnitude))
    
    # Size
    size1 = {'Y1': 100000, 'Y10': 250000}
    size2 = {'Y1': 200000, 'Y10': 500000}
    size = random_generator.integers(low=size1[tag], high=size2[tag], endpoint=True)
    
    application_size = len(application_dataset['photometry']['redshift'])
    indices = random_generator.choice(numpy.arange(application_size)[filter], size=size, replace=True, p=rate[filter] / numpy.sum(rate[filter]))
    
    # Selection
    selection_dataset = {
        'morphology': {key: application_dataset['morphology'][key][indices] for key in application_dataset['morphology'].keys()},
        'photometry': {key: application_dataset['photometry'][key][indices] for key in application_dataset['photometry'].keys()}
    }
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(dataset_folder, '{}/SOM/INFORM.pkl'.format(tag))
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    selection_size = len(selection_dataset['photometry']['redshift'])
    selection_cell_coordinate = numpy.zeros((selection_size, 2), dtype=numpy.int32)
    
    for m in range(selection_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, selection_size)
        
        if begin < stop:
            selection = {key: selection_dataset['photometry'][key][begin: stop] for key in model['usecols']}
            
            selection_column = somoclu_som._computemagcolordata(data=selection, mag_column_name=model['ref_column'], column_names=model['usecols'], colusage=model['column_usage'])
            selection_cell_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], selection_column)
    
    selection_cell_coordinate1 = selection_cell_coordinate[:, 0]
    selection_cell_coordinate2 = selection_cell_coordinate[:, 1]
    selection_cell_id = numpy.ravel_multi_index(numpy.transpose(selection_cell_coordinate), (model['n_rows'], model['n_columns']))
    
    cell_size = model['n_rows'] * model['n_columns']
    selection_cell_count = numpy.bincount(selection_cell_id, minlength=cell_size)
    selection_cell_z_true = numpy.divide(numpy.bincount(selection_cell_id, weights=selection_dataset['photometry']['redshift_true'], minlength=cell_size), selection_cell_count, out=numpy.ones(cell_size) * numpy.nan, where=selection_cell_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('size', data=size, dtype=numpy.int32)
        file['meta'].create_dataset('angle', data=angle, dtype=numpy.float32)
        file['meta'].create_dataset('color', data=color, dtype=numpy.float32)
        file['meta'].create_dataset('factor', data=factor, dtype=numpy.float32)
        file['meta'].create_dataset('fraction', data=fraction, dtype=numpy.float32)
        file['meta'].create_dataset('redshift', data=redshift, dtype=numpy.float32)
        file['meta'].create_dataset('magnitude', data=magnitude, dtype=numpy.float32)
        
        file['meta'].create_dataset('cell_size', data=cell_size, dtype=numpy.int32)
        file['meta'].create_dataset('cell_size1', data=model['n_rows'], dtype=numpy.int32)
        file['meta'].create_dataset('cell_size2', data=model['n_columns'], dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_id', data=selection_cell_id, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate1', data=selection_cell_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('cell_coordinate2', data=selection_cell_coordinate2, dtype=numpy.int32)
        
        file['meta'].create_dataset('cell_count', data=selection_cell_count, dtype=numpy.int32)
        file['meta'].create_dataset('cell_z_true', data=selection_cell_z_true, dtype=numpy.float32)
        
        file.create_group('morphology')
        for key in selection_dataset['morphology'].keys():
            file['morphology'].create_dataset(key, data=selection_dataset['morphology'][key], dtype=selection_dataset['morphology'][key].dtype)
        
        file.create_group('photometry')
        for key in selection_dataset['photometry'].keys():
            file['photometry'].create_dataset(key, data=selection_dataset['photometry'][key], dtype=selection_dataset['photometry'][key].dtype)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Selection datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of the configuration')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the selection datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the selection datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)