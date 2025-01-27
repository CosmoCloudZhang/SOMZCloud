import os
import h5py
import time
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som


def main(tag, index, folder):
    '''
    Create the degradation datasets
    
    Arguments:
        tag (str): The tag of the configuration
        index (int): The index of the degradation datasets
        folder (str): The base folder of the degradation datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    numpy.random.seed(index)
    print('Index: {}'.format(index))
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/DEGRADATION/'.format(tag)), exist_ok=True)
    
    # Application
    application_dataset = {
        'meta': {},
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        application_dataset['meta'] = {key: file['meta'][key][:].astype(numpy.float32) for key in file['meta'].keys()}
        application_dataset['morphology'] = {key: file['morphology'][key][:].astype(numpy.float32) for key in file['morphology'].keys()}
        application_dataset['photometry'] = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
    
    # Magnitude
    magnitude1 = 20
    magnitude2 = 24
    magnitude = numpy.random.uniform(low=magnitude1, high=magnitude2)
    select = (application_dataset['photometry']['mag_i_lsst'] < magnitude)
    
    # Redshift
    redshift1 = 0.5
    redshift2 = 2.0
    redshift = numpy.random.uniform(low=redshift1, high=redshift2)
    select = select & (application_dataset['photometry']['redshift'] < redshift)
    
    # Fraction
    fraction1 = 0.5
    fraction2 = 1.0
    fraction = numpy.random.uniform(low=fraction1, high=fraction2)
    
    # Label
    application_label = application_dataset['meta']['label']
    label_size = int(fraction * len(numpy.unique(application_label[select])))
    
    select_label = numpy.random.choice(numpy.unique(application_label[select]), size=label_size, replace=False)
    select = select & numpy.isin(application_label, select_label)
    
    # Size
    size1 = 100000
    size2 = 200000
    size = numpy.minimum(numpy.random.randint(low=size1, high=size2), numpy.sum(select))
    
    application_size = len(application_dataset['photometry']['redshift'])
    indices = numpy.random.choice(numpy.arange(application_size)[select], size=size, replace=False)
    
    # Degradation
    degradation_dataset = {
        'morphology': {key: application_dataset['morphology'][key][indices] for key in application_dataset['morphology'].keys()},
        'photometry': {key: application_dataset['photometry'][key][indices] for key in application_dataset['photometry'].keys()}
    }
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(som_folder, '{}/INFORM/INFORM.pkl'.format(tag))
    column_list = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    degradation_size = len(degradation_dataset['photometry']['redshift'])
    degradation_coordinate = numpy.zeros((degradation_size, 2), dtype=numpy.int32)
    
    for m in range(degradation_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, degradation_size)
        degradation = {key: degradation_dataset['photometry'][key][begin: stop].astype(numpy.float32) for key in column_list}
        
        degradation_column = somoclu_som._computemagcolordata(data=degradation, mag_column_name='mag_i_lsst', column_names=column_list, colusage='magandcolors')
        degradation_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], degradation_column)
    
    degradation_coordinate1 = degradation_coordinate[:, 0]
    degradation_coordinate2 = degradation_coordinate[:, 1]
    degradation_label = degradation_coordinate1 * model['n_columns'] + degradation_coordinate2
    
    som_size = model['n_columns'] * model['n_rows']
    degradation_count = numpy.bincount(degradation_label, minlength=som_size)
    degradation_mean = numpy.divide(numpy.bincount(degradation_label, weights=degradation_dataset['photometry']['redshift'], minlength=som_size), degradation_count, out=numpy.ones(som_size) * numpy.nan, where=degradation_count != 0)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        
        file['meta'].create_dataset('size', data=size, dtype=numpy.int32)
        file['meta'].create_dataset('fraction', data=fraction, dtype=numpy.float32)
        file['meta'].create_dataset('redshift', data=redshift, dtype=numpy.float32)
        file['meta'].create_dataset('magnitude', data=magnitude, dtype=numpy.float32)
        
        file['meta'].create_dataset('mean', data=degradation_mean, dtype=numpy.float32)
        file['meta'].create_dataset('count', data=degradation_count, dtype=numpy.int32)
        
        file['meta'].create_dataset('label', data=degradation_label, dtype=numpy.int32)
        file['meta'].create_dataset('coordinate1', data=degradation_coordinate1, dtype=numpy.int32)
        file['meta'].create_dataset('coordinate2', data=degradation_coordinate2, dtype=numpy.int32)
        
        file.create_group('morphology')
        for key in degradation_dataset['morphology'].keys():
            file['morphology'].create_dataset(key, data=degradation_dataset['morphology'][key], dtype=numpy.float32)
        
        file.create_group('photometry')
        for key in degradation_dataset['photometry'].keys():
            file['photometry'].create_dataset(key, data=degradation_dataset['photometry'][key], dtype=numpy.float32)
    
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
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the degradation datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)