import os
import h5py
import time
import numpy
import pandas
import argparse
from photerr import LsstErrorModel


def main(tag, index, folder):
    '''
    Create the augmentation datasets
    
    Arguments:
        tag (str): The tag of observing conditions
        index (int): The index of the augmentation datasets
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
    os.makedirs(os.path.join(dataset_folder, '{}/AUGMENTATION/'.format(tag)), exist_ok=True)
    
    # Catalog
    with h5py.File(os.path.join(dataset_folder, '{}/SIMULATION/SIMULATION.hdf5'.format(tag)), 'r') as file:
        catalog = {key: file[key][:].astype(numpy.float32) for key in file.keys()}
    
    # Error
    error_model = LsstErrorModel(
        nYrObs=1, 
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
    
    # AUGMENTATION
    table = error_model(pandas.DataFrame(catalog))
    
    # Redshift
    z1 = 0.5
    z2 = 2.5
    z = numpy.random.uniform(low=z1, high=z2)
    
    # Magnitude
    magnitude1 = 20
    magnitude2 = 24
    magnitude = numpy.random.uniform(low=magnitude1, high=magnitude2)
    
    # Selection
    select = (table['redshift'] < z) & (table['mag_i_lsst'] < magnitude)
    
    # Save
    with h5py.File(os.path.join(dataset_folder, '{}/AUGMENTATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
        file.create_group('photometry')
        file['photometry'].create_dataset('redshift', data=table['redshift'].values[~select])
        
        file['photometry'].create_dataset('mag_u_lsst', data=table['mag_u_lsst'].values[~select])
        file['photometry'].create_dataset('mag_g_lsst', data=table['mag_g_lsst'].values[~select])
        file['photometry'].create_dataset('mag_r_lsst', data=table['mag_r_lsst'].values[~select])
        file['photometry'].create_dataset('mag_i_lsst', data=table['mag_i_lsst'].values[~select])
        file['photometry'].create_dataset('mag_z_lsst', data=table['mag_z_lsst'].values[~select])
        file['photometry'].create_dataset('mag_y_lsst', data=table['mag_y_lsst'].values[~select])
        
        file['photometry'].create_dataset('mag_u_lsst_err', data=table['mag_u_lsst_err'].values[~select])
        file['photometry'].create_dataset('mag_g_lsst_err', data=table['mag_g_lsst_err'].values[~select])
        file['photometry'].create_dataset('mag_r_lsst_err', data=table['mag_r_lsst_err'].values[~select])
        file['photometry'].create_dataset('mag_i_lsst_err', data=table['mag_i_lsst_err'].values[~select])
        file['photometry'].create_dataset('mag_z_lsst_err', data=table['mag_z_lsst_err'].values[~select])
        file['photometry'].create_dataset('mag_y_lsst_err', data=table['mag_y_lsst_err'].values[~select])
    
    # Duration
    end = time.time()
    duration = (end - start) / 60
    
    # Return
    print('Time: {:.2f} minutes'.format(duration))
    return duration


if __name__ == '__main__':
    # Input
    PARSE = argparse.ArgumentParser(description='Augmentation datasets')
    PARSE.add_argument('--tag', type=str, required=True, help='The tag of observing conditions')
    PARSE.add_argument('--index', type=int, required=True, help='The index of the augmentation datasets')
    PARSE.add_argument('--folder', type=str, required=True, help='The base folder containing the datasets')
    
    # Argument
    TAG = PARSE.parse_args().tag
    INDEX = PARSE.parse_args().index
    FOLDER = PARSE.parse_args().folder
    
    # Output
    OUTPUT = main(TAG, INDEX, FOLDER)