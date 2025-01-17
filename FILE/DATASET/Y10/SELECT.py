import os
import h5py
import time
import numpy
import pandas
import argparse
from photerr import LsstErrorModel


def main(tag, number, folder):
    '''
    Create the selection datasets
    
    Arguments:
        tag (str): The tag of observing conditions
        number (int): The number of the selection datasets
        folder (str): The base folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/SELECTION/'.format(tag)), exist_ok=True)
    
    # Catalog
    catalog = {}
    with h5py.File(os.path.join(dataset_folder, '{}/SIMULATION/SIMULATION.hdf5'.format(tag)), 'r') as file:
        catalog['major'] = file['major'][:].astype(numpy.float32)
        catalog['minor'] = file['minor'][:].astype(numpy.float32)
        catalog['redshift'] = file['redshift'][:].astype(numpy.float32)
        catalog['mag_u_lsst'] = file['mag_u_lsst'][:].astype(numpy.float32)
        catalog['mag_g_lsst'] = file['mag_g_lsst'][:].astype(numpy.float32)
        catalog['mag_r_lsst'] = file['mag_r_lsst'][:].astype(numpy.float32)
        catalog['mag_i_lsst'] = file['mag_i_lsst'][:].astype(numpy.float32)
        catalog['mag_z_lsst'] = file['mag_z_lsst'][:].astype(numpy.float32)
        catalog['mag_y_lsst'] = file['mag_y_lsst'][:].astype(numpy.float32)
    
    # Selection
    z1 = 0.05
    z2 = 2.95
    select = (z1 < catalog['redshift']) & (catalog['redshift'] < z2)
    
    magnitude1 = 15
    magnitude2 = 30
    select = select & (magnitude1 < catalog['mag_i_lsst']) & (catalog['mag_i_lsst'] < magnitude2)
    
    for key in catalog:
        catalog[key] = catalog[key][select]
    
    # Error
    error_model = LsstErrorModel(
        nYrObs=10, 
        sigLim=3.0,
        absFlux=True,
        ndMode='sigLim', 
        majorCol='major', 
        minorCol='minor', 
        decorrelate=True,
        extendedType='auto',
        renameDict={
            'u': 'mag_u_lsst',
            'g': 'mag_g_lsst',
            'r': 'mag_r_lsst',
            'i': 'mag_i_lsst',
            'z': 'mag_z_lsst',
            'y': 'mag_y_lsst'
        }
    )
    
    # Selection
    for index in range(1, number + 1):
        print('Index: {:.0f}'.format(index))
        
        table = error_model(pandas.DataFrame(catalog))
        
        # Save
        with h5py.File(os.path.join(dataset_folder, '{}/SELECTION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('photometry')
            file['photometry'].create_dataset('redshift', data=table['redshift'].values)
            
            file['photometry'].create_dataset('mag_u_lsst', data=table['mag_u_lsst'].values)
            file['photometry'].create_dataset('mag_g_lsst', data=table['mag_g_lsst'].values)
            file['photometry'].create_dataset('mag_r_lsst', data=table['mag_r_lsst'].values)
            file['photometry'].create_dataset('mag_i_lsst', data=table['mag_i_lsst'].values)
            file['photometry'].create_dataset('mag_z_lsst', data=table['mag_z_lsst'].values)
            file['photometry'].create_dataset('mag_y_lsst', data=table['mag_y_lsst'].values)
            
            file['photometry'].create_dataset('mag_u_lsst_err', data=table['mag_u_lsst_err'].values)
            file['photometry'].create_dataset('mag_g_lsst_err', data=table['mag_g_lsst_err'].values)
            file['photometry'].create_dataset('mag_r_lsst_err', data=table['mag_r_lsst_err'].values)
            file['photometry'].create_dataset('mag_i_lsst_err', data=table['mag_i_lsst_err'].values)
            file['photometry'].create_dataset('mag_z_lsst_err', data=table['mag_z_lsst_err'].values)
            file['photometry'].create_dataset('mag_y_lsst_err', data=table['mag_y_lsst_err'].values)
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Selection Datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of observing conditions')
    PARSE.add_argument('--number', type=int, required=True, help='The number of the selection datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    NUMBER = PARSE.parse_args().number
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, NUMBER, FOLDER)