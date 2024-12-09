import os
import h5py
import time
import numpy
import pandas
import argparse
from photerr import LsstErrorModel

def selection(index, directory):
    '''
    Create individual selection datasets
    
    Arguments:
        catalog (dict): The catalog data
        select (numpy.ndarray): The selection of the catalog
        band_list (list): The list of bands for LSST
    
    Returns:
        data (dict): The selection dataset
    '''
    # Catalog
    catalog = {}
    catalog_name = os.path.join(directory, 'training_samples/training_sample{}.hdf5'.format(index - 1))
    with h5py.File(catalog_name, 'r') as file:
        for key, value in file['photometry'].items():
            catalog[key] = value[...].astype(numpy.float32)
    
    # Error
    error_model = LsstErrorModel(
        nYrObs=1, 
        sigLim=3.0,
        absFlux=True,
        ndMode='sigLim', 
        decorrelate=True,
        extendedType='point', 
        renameDict={
            'u': 'mag_u_lsst',
            'g': 'mag_g_lsst',
            'r': 'mag_r_lsst',
            'i': 'mag_i_lsst',
            'z': 'mag_z_lsst',
            'y': 'mag_y_lsst'
        }
    )
    
    table = error_model(pandas.DataFrame(catalog))
    
    # Band
    band_list = ['u_lsst', 'g_lsst', 'r_lsst', 'i_lsst', 'z_lsst', 'y_lsst']
    for band in band_list:
        
        # Photometry
        mag = table['mag_{}'.format(band)].values
        mag_err = table['mag_{}_err'.format(band)].values
        
        catalog['mag_{}'.format(band)] = mag
        catalog['mag_err_{}'.format(band)] = mag_err
        catalog['snr_{}'.format(band)] = 2.5 / numpy.log(10) / mag_err
    
    # Redshift
    z1 = 0.0
    z2 = 3.0
    select = (z1 < catalog['redshift']) & (catalog['redshift'] < z2)
    
    # Magnitude
    mag1 = 16.0
    mag2 = 26.0
    select = select & (mag1 < catalog['mag_i_lsst']) & (catalog['mag_i_lsst'] < mag2)
    
    # SNR
    snr1 = 5.0
    snr2 = 5.0
    select = select & (snr1 < catalog['snr_r_lsst']) & (catalog['snr_i_lsst'] > snr2)
    
    # Data
    data = {}
    data['redshift'] = catalog['redshift'][select]
    
    for band in band_list:
        data['mag_{}'.format(band)] = catalog['mag_{}'.format(band)][select]
        data['mag_err_{}'.format(band)] = catalog['mag_err_{}'.format(band)][select]
    
    # Save
    return data


def main(number, folder, directory):
    '''
    Create the selection datasets
    
    Arguments:
        number (int): The number of datasets
        folder (str): The base folder of the catalog
        directory (str): The base directory of the catalog
    
    Returns:
        duration (float): The duration of the process
    '''
    # Start
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Index
    for index in range(1, number + 1):
        print('Index: {:.0f}'.format(index))
        
        # Data
        data = selection(index, directory)
        
        # Save
        os.makedirs(dataset_folder, exist_ok=True)
        os.makedirs(os.path.join(dataset_folder, 'SELECTION'), exist_ok=True)
        with h5py.File(os.path.join(dataset_folder, 'SELECTION/DATA{:.0f}.hdf5'.format(index)), 'w') as file:
            file.create_group('photometry')
            
            for key, value in data.items():
                file['photometry'].create_dataset(key, data=value)
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Selection dataset')
    PARSE.add_argument('--number', type=int, required=True, help='The number of datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the catalog')
    PARSE.add_argument('--directory', type=str, required=True, help='The base directory of the catalog')
    
    # Argument
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    DIRECTORY = PARSE.parse_args().directory
    
    # Output
    OUTPUT = main(NUMBER, FOLDER, DIRECTORY)