import os
import h5py
import time
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som


def main(tag, number, folder):
    '''
    Create the augmentation datasets
    
    Arguments:
        tag (str): The tag of the configuration
        number (int): The number of the augmentation datasets
        folder (str): The base folder of the augmentation datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/AUGMENTATION/'.format(tag)), exist_ok=True)
    
    # SOM
    data_store = core.stage.RailStage.data_store
    data_store.__class__.allow_overwrite = True
    
    model_name = os.path.join(som_folder, '{}/INFORM/INFORM.pkl'.format(tag))
    column_list = ['mag_u_lsst', 'mag_g_lsst', 'mag_r_lsst', 'mag_i_lsst', 'mag_z_lsst', 'mag_y_lsst']
    model = data_store.read_file(key='model', path=model_name, handle_class=core.data.ModelHandle)()
    
    # Loop
    numpy.random.seed(0)
    for index in range(1, number + 1):
        print('Index: {}'.format(index))
        
        # Selection
        selection_dataset = {
            'morphology': {},
            'photometry': {}
        }
        
        with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            selection_dataset['morphology'] = {key: file['morphology'][key][:].astype(numpy.float32) for key in file['morphology'].keys()}
            selection_dataset['photometry'] = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        
        # Degradation
        degradation_dataset = {
            'morphology': {},
            'photometry': {}
        }
        
        with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            degradation_dataset['morphology'] = {key: file['morphology'][key][:].astype(numpy.float32) for key in file['morphology'].keys()}
            degradation_dataset['photometry'] = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        
        # Redshift
        redshift = numpy.max(degradation_dataset['photometry']['redshift'])
        filter = selection_dataset['photometry']['redshift'] > redshift
        
        # Magnitude
        magnitude = numpy.max(degradation_dataset['photometry']['mag_i_lsst'])
        filter = filter | (selection_dataset['photometry']['mag_i_lsst'] > magnitude)
        
        # Fraction
        fraction1 = 0.5
        fraction2 = 1.0
        fraction = numpy.random.uniform(fraction1, fraction2)
        size = int(fraction * len(degradation_dataset['photometry']['redshift']))
        indices = numpy.random.choice(numpy.arange(len(selection_dataset['photometry']['redshift']))[filter], size=size, replace=False)
        
        for key in selection_dataset['morphology'].keys():
            selection_dataset['morphology'][key] = selection_dataset['morphology'][key][indices]
        
        for key in selection_dataset['photometry'].keys():
            selection_dataset['photometry'][key] = selection_dataset['photometry'][key][indices]
        
        # Degradation SOM
        degradation_column = somoclu_som._computemagcolordata(data=degradation_dataset['photometry'], mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
        degradation_coordinate = somoclu_som.get_bmus(model['som'], degradation_column)
        
        degradation_coordinate1 = degradation_coordinate[:, 0]
        degradation_coordinate2 = degradation_coordinate[:, 1]
        
        degradation_label = degradation_coordinate1 * model['n_columns'] + degradation_coordinate2
        degradation_occupation = numpy.bincount(degradation_label, minlength=model['n_rows'] * model['n_columns'])
        
        # Threshold
        threshold1 = 0
        threshold2 = 10
        threshold = numpy.random.uniform(threshold1, threshold2)
        filter_label = numpy.arange(model['n_rows'] * model['n_columns'])[degradation_occupation < threshold]
        
        # Selection SOM
        selection_column = somoclu_som._computemagcolordata(data=selection_dataset['photometry'], mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
        selection_coordinate = somoclu_som.get_bmus(model['som'], selection_column)
        
        selection_coordinate1 = selection_coordinate[:, 0]
        selection_coordinate2 = selection_coordinate[:, 1]
        
        selection_label = selection_coordinate1 * model['n_columns'] + selection_coordinate2
        filter = numpy.isin(selection_label, filter_label)
        
        coordinate1 = selection_coordinate1[filter]
        coordinate2 = selection_coordinate2[filter]
        label = selection_label[filter]
        
        # Save
        with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('meta')
            file['meta'].create_dataset('fraction', data=fraction)
            file['meta'].create_dataset('redshift', data=redshift)
            file['meta'].create_dataset('magnitude', data=magnitude)
            file['meta'].create_dataset('threshold', data=threshold)
            
            file['meta'].create_dataset('label', data=label)
            file['meta'].create_dataset('coordinate1', data=coordinate1)
            file['meta'].create_dataset('coordinate2', data=coordinate2)
            
            file.create_group('morphology')
            for key in selection_dataset['morphology'].keys():
                file['morphology'].create_dataset(key, data=selection_dataset['morphology'][key][filter], dtype=numpy.float32)
            
            file.create_group('photometry')
            for key in selection_dataset['photometry'].keys():
                file['photometry'].create_dataset(key, data=selection_dataset['photometry'][key][filter], dtype=numpy.float32)
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
    PARSE.add_argument('--number', type=int, required=True, help='The number of the augmentation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the augmentation datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)