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
        tag (str): The tag of observing conditions
        number (int): The number of the augmentation datasets
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
        
        # Catalog
        with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            catalog = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        length = len(catalog['redshift'])
        
        with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            meta = {key: file['meta'][key][:].astype(numpy.float32) for key in file['meta'].keys()}
            table = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        count = len(table['redshift'])
        
        # Redshift
        redshift = meta['redshift']
        filter = catalog['redshift'] > redshift
        
        # Magnitude
        magnitude = meta['magnitude']
        filter = filter | (catalog['mag_i_lsst'] > magnitude)
        
        # Fraction
        fraction = count / numpy.sum(filter)
        indices = numpy.arange(length)[filter][numpy.random.uniform(low=0, high=1, size=numpy.sum(filter)) > fraction]
        filter[indices] = False
        
        for key in catalog.keys():
            catalog[key] = catalog[key][filter]
        
        # Table SOM
        table_column = somoclu_som._computemagcolordata(data=table, mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
        table_coordinate = somoclu_som.get_bmus(model['som'], table_column)
        
        table_coordinate1 = table_coordinate[:, 0]
        table_coordinate2 = table_coordinate[:, 1]
        
        table_label = table_coordinate1 * model['n_columns'] + table_coordinate2
        table_occupation = numpy.bincount(table_label, minlength=model['n_rows'] * model['n_columns'])
        
        # Catalog SOM
        catalog_column = somoclu_som._computemagcolordata(data=catalog, mag_column_name='mag_i_lsst', column_names=column_list, colusage='colors')
        catalog_coordinate = somoclu_som.get_bmus(model['som'], catalog_column)
        
        catalog_coordinate1 = catalog_coordinate[:, 0]
        catalog_coordinate2 = catalog_coordinate[:, 1]
        
        catalog_label = catalog_coordinate1 * model['n_columns'] + catalog_coordinate2
        
        select_label = table_label[table_occupation < table_occupation.max() / 2]
        
        select = numpy.isin(catalog_coordinate1 * model['n_columns'] + catalog_coordinate2, select_label)
        
        coordinate1 = catalog_coordinate1[select]
        coordinate2 = catalog_coordinate2[select]
        label = coordinate1 * model['n_columns'] + coordinate2
        
        # Save
        with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('meta')
            file['meta'].create_dataset('fraction', data=fraction)
            file['meta'].create_dataset('redshift', data=redshift)
            file['meta'].create_dataset('magnitude', data=magnitude)
            
            file['meta'].create_dataset('label', data=label)
            file['meta'].create_dataset('coordinate1', data=coordinate1)
            file['meta'].create_dataset('coordinate2', data=coordinate2)
            
            file.create_group('photometry')
            file['photometry'].create_dataset('redshift', data=catalog['redshift'][select])
            
            file['photometry'].create_dataset('mag_u_lsst', data=catalog['mag_u_lsst'][select])
            file['photometry'].create_dataset('mag_g_lsst', data=catalog['mag_g_lsst'][select])
            file['photometry'].create_dataset('mag_r_lsst', data=catalog['mag_r_lsst'][select])
            file['photometry'].create_dataset('mag_i_lsst', data=catalog['mag_i_lsst'][select])
            file['photometry'].create_dataset('mag_z_lsst', data=catalog['mag_z_lsst'][select])
            file['photometry'].create_dataset('mag_y_lsst', data=catalog['mag_y_lsst'][select])
            
            file['photometry'].create_dataset('mag_u_lsst_err', data=catalog['mag_u_lsst_err'][select])
            file['photometry'].create_dataset('mag_g_lsst_err', data=catalog['mag_g_lsst_err'][select])
            file['photometry'].create_dataset('mag_r_lsst_err', data=catalog['mag_r_lsst_err'][select])
            file['photometry'].create_dataset('mag_i_lsst_err', data=catalog['mag_i_lsst_err'][select])
            file['photometry'].create_dataset('mag_z_lsst_err', data=catalog['mag_z_lsst_err'][select])
            file['photometry'].create_dataset('mag_y_lsst_err', data=catalog['mag_y_lsst_err'][select])
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of observing conditions')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the augmentation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)