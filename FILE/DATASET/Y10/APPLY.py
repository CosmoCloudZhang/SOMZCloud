import os
import h5py
import time
import numpy
import pandas
import argparse
from photerr import LsstErrorModel


def main(tag, index, folder):
    '''
    Create the application datasets
    
    Arguments:
        tag (str): The tag of observing conditions
        index (int): The index of the application datasets
        folder (str): The base folder containing the datasets
    
    Returns:
        duration (float): The duration of the process
    '''
    # Path
    start = time.time()
    print('Index: {:.0f}'.format(index))
    
    # Path
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/APPLICATION/'.format(tag)), exist_ok=True)
    
    # Catalog
    with h5py.File(os.path.join(dataset_folder, '{}/OBSERVATION/OBSERVATION.hdf5'.format(tag)), 'r') as file:
        catalog = {key: file[key][:].astype(numpy.float32) for key in file.keys()}
    
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
    
    # Application
    table = error_model(pandas.DataFrame(catalog))
    select = numpy.random.choice(len(table), len(table), replace=True)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=table['redshift'].values[select])
        
        file['photometry'].create_dataset('mag_u_lsst', data=table['mag_u_lsst'].values[select])
        file['photometry'].create_dataset('mag_g_lsst', data=table['mag_g_lsst'].values[select])
        file['photometry'].create_dataset('mag_r_lsst', data=table['mag_r_lsst'].values[select])
        file['photometry'].create_dataset('mag_i_lsst', data=table['mag_i_lsst'].values[select])
        file['photometry'].create_dataset('mag_z_lsst', data=table['mag_z_lsst'].values[select])
        file['photometry'].create_dataset('mag_y_lsst', data=table['mag_y_lsst'].values[select])
        
        file['photometry'].create_dataset('mag_u_lsst_err', data=table['mag_u_lsst_err'].values[select])
        file['photometry'].create_dataset('mag_g_lsst_err', data=table['mag_g_lsst_err'].values[select])
        file['photometry'].create_dataset('mag_r_lsst_err', data=table['mag_r_lsst_err'].values[select])
        file['photometry'].create_dataset('mag_i_lsst_err', data=table['mag_i_lsst_err'].values[select])
        file['photometry'].create_dataset('mag_z_lsst_err', data=table['mag_z_lsst_err'].values[select])
        file['photometry'].create_dataset('mag_y_lsst_err', data=table['mag_y_lsst_err'].values[select])
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Application dataset')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of observing conditions')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the application datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)