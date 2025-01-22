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
    # Path
    start = time.time()
    numpy.random.seed(index)
    print('Index: {}'.format(index))
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/AUGMENTATION/'.format(tag)), exist_ok=True)
    
    # Degradation
    with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        degradation_occupation = file['meta']['occupation'][:].astype(numpy.float32)
        degradation_redshift = file['photometry']['redshift'][:].astype(numpy.float32)
        degradation_magnitude = file['photometry']['mag_i_lsst'][:].astype(numpy.float32)
    degradation_size = len(degradation_redshift)
    
    # Selection
    selection_dataset = {
        'meta': {},
        'morphology': {},
        'photometry': {}
    }
    
    with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
        selection_dataset['meta'] = {key: file['meta'][key][:].astype(numpy.float32) for key in file['meta'].keys()}
        selection_dataset['morphology'] = {key: file['morphology'][key][:].astype(numpy.float32) for key in file['morphology'].keys()}
        selection_dataset['photometry'] = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
    
    selection_label = selection_dataset['meta']['label']
    selection_size = len(selection_dataset['photometry']['redshift'])
    filter = numpy.isin(selection_label, numpy.arange(len(degradation_occupation))[degradation_occupation == 0])
    
    # Redshift
    redshift = numpy.max(degradation_redshift)
    filter = filter | (selection_dataset['photometry']['redshift'] > redshift)
    
    # Magnitude
    magnitude = numpy.max(degradation_magnitude)
    filter = filter | (selection_dataset['photometry']['mag_i_lsst'] > magnitude)
    
    # Choice
    fraction = numpy.sum(degradation_occupation > 0) / len(degradation_occupation)
    indices = numpy.random.choice(numpy.arange(selection_size)[filter], size=int((1 - fraction) * degradation_size), replace=False)
    
    # Augmentation
    augmentation_dataset = {
        'meta': {'fraction': fraction, 'redshift': redshift, 'magnitude': magnitude},
        'morphology': {key: selection_dataset['morphology'][key][indices] for key in selection_dataset['morphology'].keys()},
        'photometry': {key: selection_dataset['photometry'][key][indices] for key in selection_dataset['photometry'].keys()}
    }
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(som_folder, '{}/INFORM/INFORM.pkl'.format(tag))
    column_list = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    chunk = 100000
    augmentation_size = len(augmentation_dataset['photometry']['redshift'])
    augmentation_coordinate = numpy.zeros((augmentation_size, 2), dtype=numpy.int32)
    
    for m in range(augmentation_size // chunk + 1):
        begin = m * chunk
        stop = min((m + 1) * chunk, augmentation_size)
        augmentation = {key: augmentation_dataset['photometry'][key][begin: stop].astype(numpy.float32) for key in column_list}
        
        augmentation_column = somoclu_som._computemagcolordata(data=augmentation, mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
        augmentation_coordinate[begin: stop, :] = somoclu_som.get_bmus(model['som'], augmentation_column)
    
    augmentation_coordinate1 = augmentation_coordinate[:, 0]
    augmentation_coordinate2 = augmentation_coordinate[:, 1]
    augmentation_label = augmentation_coordinate1 * model['n_columns'] + augmentation_coordinate2
    
    augmentation_occupation = numpy.bincount(augmentation_label, minlength=model['n_rows'] * model['n_columns'])
    augmentation_mean = numpy.divide(numpy.bincount(augmentation_label, weights=augmentation_dataset['photometry']['redshift'], minlength=model['n_rows'] * model['n_columns']), augmentation_occupation, out=numpy.ones(model['n_rows'] * model['n_columns']) * numpy.nan, where=augmentation_occupation != 0)
    
    augmentation_dataset['meta']['label'] = augmentation_label
    augmentation_dataset['meta']['coordinate1'] = augmentation_coordinate1
    augmentation_dataset['meta']['coordinate2'] = augmentation_coordinate2
    
    augmentation_dataset['meta']['mean'] = augmentation_mean
    augmentation_dataset['meta']['occupation'] = augmentation_occupation
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('meta')
        for key in augmentation_dataset['meta'].keys():
            file['meta'].create_dataset(key, data=augmentation_dataset['meta'][key], dtype=numpy.float32)
        
        file.create_group('morphology')
        for key in selection_dataset['morphology'].keys():
            file['morphology'].create_dataset(key, data=augmentation_dataset['morphology'][key], dtype=numpy.float32)
        
        file.create_group('photometry')
        for key in selection_dataset['photometry'].keys():
            file['photometry'].create_dataset(key, data=augmentation_dataset['photometry'][key], dtype=numpy.float32)
    
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