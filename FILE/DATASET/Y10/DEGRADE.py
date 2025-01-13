import os
import h5py
import time
import numpy
import argparse


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
    dataset_folder = os.path.join(folder, 'DATASET/')
    os.makedirs(os.path.join(dataset_folder, '{}'.format(tag)), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, '{}/DEGRADATION/'.format(tag)), exist_ok=True)
    
    for index in range(1, number + 1):
        print('Index: {}'.format(index))
        
        # Catalog
        with h5py.File(os.path.join(dataset_folder, '{}/APPLICATION/DATA{}.hdf5'.format(tag, index)), 'r') as file:
            catalog = {key: file['photometry'][key][:].astype(numpy.float32) for key in file['photometry'].keys()}
        
        # Redshift
        z1 = 0.5
        z2 = 2.0
        z = numpy.random.uniform(low=z1, high=z2)
        
        # Magnitude
        magnitude1 = 20
        magnitude2 = 24
        magnitude = numpy.random.uniform(low=magnitude1, high=magnitude2)
        
        # SOM Coordinates
        coordinate1 = numpy.random.randint(low=0, high=100)
        coordinate2 = numpy.random.randint(low=0, high=100)
        
        # Selection
        select = (catalog['redshift'] < z) & (catalog['mag_i_lsst'] < magnitude)
        print(z, magnitude, numpy.sum(select))
        
        # Save
        with h5py.File(os.path.join(dataset_folder, '{}/DEGRADATION/DATA{}.hdf5'.format(tag, index)), 'w') as file:
            file.create_group('meta')
            file['meta'].create_dataset('z', data=z)
            file['meta'].create_dataset('magnitude', data=magnitude)
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