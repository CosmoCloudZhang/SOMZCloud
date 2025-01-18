import os
import h5py
import time
import numpy
import argparse
from rail import core
from rail.estimation.algos import somoclu_som

def main(tag, number, folder):
    '''
    Create the degradation datasets
    
    Arguments:
        tag (str): The tag of observing conditions
        number (int): The number of the degradation datasets
        folder (str): The base folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    
    # Path
    som_folder = os.path.join(folder, 'SOM/')
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/DEGRADATION/'.format(tag)), exist_ok=True)
    
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
        
        # Catalog
        catalog = {
            'morphology': {},
            'photometry': {}
        }
        
        with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            catalog['morphology'] = {key: file['morphology'][key][:].astype(numpy.float32) for key in file['morphology'].keys()}
            catalog['photometry'] = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        
        # Redshift
        redshift1 = 1.0
        redshift2 = 2.5
        redshift = numpy.random.uniform(low=redshift1, high=redshift2)
        filter = (catalog['photometry']['redshift'] < redshift)
        
        # Magnitude
        magnitude1 = 21
        magnitude2 = 25
        magnitude = numpy.random.uniform(low=magnitude1, high=magnitude2)
        filter = filter & (catalog['photometry']['mag_i_lsst'] < magnitude)
        
        # Fraction
        count = 500000
        fraction = count / numpy.sum(filter)
        indices = numpy.arange(len(catalog['photometry']['redshift']))[filter]
        indices = indices[numpy.random.uniform(low=0, high=1, size=numpy.sum(filter)) > fraction]
        filter[indices] = False
        
        # Filter
        for key in catalog['morphology'].keys():
            catalog['morphology'][key] = catalog['morphology'][key][filter]
        
        for key in catalog['photometry'].keys():
            catalog['photometry'][key] = catalog['photometry'][key][filter]
        
        # SOM
        catalog_column = somoclu_som._computemagcolordata(data=catalog['photometry'], mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
        catalog_coordinate = somoclu_som.get_bmus(model['som'], catalog_column)
        
        catalog_coordinate1 = catalog_coordinate[:, 0]
        catalog_coordinate2 = catalog_coordinate[:, 1]
        
        catalog_label = numpy.unique(catalog_coordinate1 * model['n_columns'] + catalog_coordinate2)
        select_label = numpy.random.choice(catalog_label, size=catalog_label.size // 2, replace=False)
        select = numpy.isin(catalog_coordinate1 * model['n_columns'] + catalog_coordinate2, select_label)
        
        coordinate1 = catalog_coordinate1[select]
        coordinate2 = catalog_coordinate2[select]
        label = coordinate1 * model['n_columns'] + coordinate2
        
        # Save
        with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('meta')
            file['meta'].create_dataset('fraction', data=fraction, dtype=numpy.float32)
            file['meta'].create_dataset('redshift', data=redshift, dtype=numpy.float32)
            file['meta'].create_dataset('magnitude', data=magnitude, dtype=numpy.float32)
            
            file['meta'].create_dataset('label', data=label, dtype=numpy.int32)
            file['meta'].create_dataset('coordinate1', data=coordinate1, dtype=numpy.int32)
            file['meta'].create_dataset('coordinate2', data=coordinate2, dtype=numpy.int32)
            
            file.create_group('morphology')
            for key in catalog['morphology'].keys():
                file['morphology'].create_dataset(key, data=catalog['morphology'][key][select], dtype=numpy.float32)
            
            file.create_group('photometry')
            for key in catalog['photometry'].keys():
                file['photometry'].create_dataset(key, data=catalog['photometry'][key][select], dtype=numpy.float32)
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Degradation datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of observing conditions')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the degradation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)