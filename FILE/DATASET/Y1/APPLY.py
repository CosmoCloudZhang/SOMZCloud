import os
import h5py
import time
import numpy
import pandas
import argparse
from photerr import LsstErrorModel


def main(tag, number, folder):
    '''
    Create the application datasets
    
    Arguments:
        tag (str): The tag of observing conditions
        number (int): The number of the application datasets
        folder (str): The base folder containing the catalog
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    
    # Catalog
    with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/OBSERVATION.hdf5'.format(tag)), 'r') as file:
        catalog = {key: file[key][:].astype(numpy.float32) for key in file.keys()}
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
    # Application
    for index in range(1, number + 1):
        print('Index: {:.0f}'.format(index))
        

    
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
    
    # Select
    width = 4000000
    length = len(catalog['redshift'][select])
    indices = numpy.random.choice(length, width, replace=True)
    
    # Data
    data = {}
    data['redshift'] = catalog['redshift'][select][indices]
    
    for band in band_list:
        data['mag_{}'.format(band)] = catalog['mag_{}'.format(band)][select][indices]
        data['mag_err_{}'.format(band)] = catalog['mag_err_{}'.format(band)][select][indices]
    
    # Save
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'APPLICATION'), exist_ok=True)
    
    with h5py.File(os.path.join(dataset_folder, 'APPLICATION/DATA{:.0f}.hdf5'.format(index)), 'w') as file:
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
    PARSE = argparse.ArgumentParser(description='Application dataset')
    PARSE.add_argument('--number', type=int, required=True, help='The number of datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder of the catalog')
    
    # Argument
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(NUMBER, FOLDER)